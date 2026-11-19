from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel
from torch import Tensor

from lib.dataloaders.batches import ShuffledRowBatches
from lib.dataloaders.common import d4
from lib.train.lightning import single_device_trainer
from lib.utils.log import log_event
from lib.utils.torch_utils import configure_torch
from lib.utils.utils import check, load_yaml_as, pick_device
from photoz.artifacts import HISTORY_NAME as HISTORY_NAME
from photoz.data.euclid_packet import (
    CLIP_NORM,
    KEYS,
    LR,
    SSL_BATCH,
    SSL_EPOCHS,
    WD,
    Packet,
    Rows,
    reset_rng,
    state_copy,
)
from photoz.models.euclid_clip import CLIPModel, ITNetwork
from photoz.training.clip_epochs import (
    ClipTrainingEpoch,
    OptimizerSchedule,
    scheduled_adamw,
)

STATE_NAME = "clip_state.pt"
OLD_IMAGES = "old_images_f16.npy"
OLD_ARRAYS = "old_arrays.npz"
STORE_ROWS = "rows.npz"
STORE_SHARDS = "shards.json"
NEW_KEYS = ("image_available", "table_values", "table_mask", "table_available")


class EuclidClipPretrainConfig(BaseModel):
    seed: int
    epochs: int = SSL_EPOCHS
    packet: Path
    stage_dir: Path
    store_dir: Path
    shard_dir: Path | None = None
    out_dir: Path
    device: str = "auto"
    num_threads: int = 4
    memory_fraction: float = 0.3


class BatchSource(Protocol):
    def batch(
        self, rows: Rows, device: torch.device, spectra: bool = True
    ) -> dict[str, Tensor]: ...


def shard_path(raw: str, store_dir: Path, shard_dir: Path | None) -> Path:
    path = Path(raw)
    if shard_dir is not None:
        return shard_dir / path.name
    return path if path.is_absolute() else store_dir / path


class UnionStore:
    def __init__(
        self,
        stage_dir: Path,
        store_dir: Path,
        rows_new: Rows,
        descriptor: NDArray[np.floating],
        shard_dir: Path | None = None,
    ) -> None:
        self.old_images: NDArray[np.float16] = np.load(stage_dir / OLD_IMAGES, mmap_mode="r")
        with np.load(stage_dir / OLD_ARRAYS) as old:
            self.old: dict[str, NDArray[Any]] = {k: old[k] for k in KEYS if k != "image"}
            old_ids, old_fields = old["object_id"], old["field_code"]
        self.n_old = len(old_ids)
        self.rows_new = np.asarray(rows_new, np.int64)
        specs: list[dict[str, Any]] = json.loads((store_dir / STORE_SHARDS).read_text())
        self.shards: list[NDArray[np.float16]] = [
            np.memmap(
                shard_path(str(s["path"]), store_dir, shard_dir),
                dtype=np.float16,
                mode="r",
                offset=int(s["offset"]),
                shape=tuple(int(v) for v in s["shape"]),
            )
            for s in specs
        ]
        with np.load(store_dir / STORE_ROWS) as store:
            self.shard: NDArray[np.integer] = store["shard"][self.rows_new]
            self.local: NDArray[np.integer] = store["local"][self.rows_new]
            self.new: dict[str, NDArray[Any]] = {k: store[k][self.rows_new] for k in NEW_KEYS}
            new_ids, new_fields = store["object_id"], store["field_code"]
        self.ids: NDArray[np.int64] = np.r_[old_ids, new_ids[self.rows_new]]
        self.fields: NDArray[np.integer] = np.r_[old_fields, new_fields[self.rows_new]]
        self.descriptor = torch.as_tensor(descriptor)

    def images(self, ra: NDArray[np.int64], rb: NDArray[np.int64]) -> NDArray[np.float32]:
        out = np.empty((len(ra) + len(rb),) + self.old_images.shape[1:], np.float32)
        o = np.argsort(ra)
        out[o] = self.old_images[ra[o]]
        sh, lo = self.shard[rb], self.local[rb]
        for s in np.unique(sh):
            m = np.flatnonzero(sh == s)
            m = m[np.argsort(lo[m])]
            out[len(ra) + m] = self.shards[int(s)][lo[m]]
        return out

    def batch(self, rows: Rows, device: torch.device, spectra: bool = True) -> dict[str, Tensor]:
        r = np.asarray(rows)
        a, b = r < self.n_old, r >= self.n_old
        ra, rb = r[a], r[b] - self.n_old
        perm = np.r_[np.flatnonzero(a), np.flatnonzero(b)]
        image = np.empty((len(r),) + self.old_images.shape[1:], np.float32)
        image[perm] = self.images(ra, rb)
        out = {"image": torch.from_numpy(image).to(device, non_blocking=True)}
        for k in KEYS:
            if k == "image" or (not spectra and k.startswith("spectrum")):
                continue
            src = self.old[k]
            arr = np.zeros((len(r),) + src.shape[1:], src.dtype)
            arr[a] = src[ra]
            if b.any() and k in self.new:
                arr[b] = self.new[k][rb]
            out[k] = torch.from_numpy(arr).to(device, non_blocking=True)
        out["table_descriptor"] = self.descriptor.to(device)
        out["field_code"] = torch.as_tensor(self.fields[r])
        out["object_id"] = torch.as_tensor(self.ids[r])
        return out


class ContrastivePretraining(ClipTrainingEpoch):
    def __init__(
        self, clip: CLIPModel, source: BatchSource, seed: int, epochs: int, device: torch.device
    ) -> None:
        super().__init__(epochs)
        self.clip = clip
        self.source = source
        self.seed = seed
        self.aug = torch.Generator(device=device).manual_seed(seed + 2000)
        self.pairs: dict[str, int] = {}
        self.t0 = 0.0

    def configure_optimizers(self) -> OptimizerSchedule:
        return scheduled_adamw(self.clip.ssl_parameters(), self.epochs, lr=LR, weight_decay=WD)

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.pairs.clear()
        self.t0 = time.monotonic()

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        rows = batch.numpy()
        data = self.source.batch(rows, self.device)
        codes = torch.randint(8, (len(rows),), generator=self.aug, device=self.device)
        data["image"] = d4(data["image"], codes)
        loss: Tensor
        loss, counts = self.clip(data)
        check(torch.isfinite(loss), "finite CLIP loss")
        self.record_loss(loss, len(rows))
        for key, count in counts.items():
            self.pairs[key] = self.pairs.get(key, 0) + count
        return loss

    def on_train_epoch_end(self) -> None:
        record = {
            "epoch": self.current_epoch + 1,
            "lr": self.lr,
            "loss": float(self.trainer.callback_metrics["train/loss"]),
            "examples": self.examples,
            "pair_examples": dict(self.pairs),
            "seconds": time.monotonic() - self.t0,
        }
        self.history.append(record)
        log_event(phase="ssl", seed=self.seed, **record)


def pretrain(
    it: ITNetwork,
    source: BatchSource,
    rows: NDArray[np.int64],
    seed: int,
    epochs: int,
    device: torch.device,
) -> tuple[CLIPModel, list[dict[str, Any]]]:
    reset_rng(seed + 1000)
    clip = CLIPModel(it).to(device)
    module = ContrastivePretraining(clip, source, seed, epochs, device)
    batches = ShuffledRowBatches(rows, seed + 1000, SSL_BATCH, drop_last=True)
    single_device_trainer(device, epochs, gradient_clip=CLIP_NORM).fit(module, batches.loader())
    return clip.to(device), module.history


def run_train(config_path: Path) -> None:
    cfg = load_yaml_as(str(config_path), EuclidClipPretrainConfig)
    device = pick_device(cfg.device)
    configure_torch(cfg.num_threads, cfg.memory_fraction, device)
    state_path = cfg.out_dir / STATE_NAME
    check(not state_path.exists(), f"exclusive output: {state_path}")
    t0 = time.monotonic()
    packet = Packet(cfg.packet)
    descriptor = packet.descriptor
    packet.close()
    with np.load(cfg.store_dir / STORE_ROWS) as store:
        keep = np.flatnonzero(store["ssl_keep"])
    union = UnionStore(cfg.stage_dir, cfg.store_dir, keep, descriptor, cfg.shard_dir)
    log_event(
        phase="loaded_1m",
        old=int(union.n_old),
        new=int(len(keep)),
        seconds=round(time.monotonic() - t0, 1),
    )
    reset_rng(cfg.seed)
    it = ITNetwork().to(device)
    clip, history = pretrain(it, union, np.arange(len(union.ids)), cfg.seed, cfg.epochs, device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state_copy(clip), state_path)
    (cfg.out_dir / HISTORY_NAME).write_text(json.dumps(history, indent=1) + "\n")
    log_event(phase="pretrain_done", seed=cfg.seed, seconds=round(time.monotonic() - t0, 1))
