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

from lib.dataloaders.euclid_packet import (
    CLIP_NORM,
    KEYS,
    LR,
    SSL_BATCH,
    SSL_EPOCHS,
    WD,
    Packet,
    Rows,
    d4,
    reset_rng,
    schedule_lr,
    state_copy,
)
from lib.models.euclid_clip import CLIPModel, ITNetwork
from lib.utils.log import log_event
from lib.utils.torch_utils import configure_torch
from lib.utils.utils import check, load_yaml_as, pick_device

STATE_NAME = "clip_state.pt"
HISTORY_NAME = "ssl_history.json"
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
    params = clip.ssl_parameters()
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    rng = np.random.default_rng(seed + 1000)
    aug = torch.Generator(device=device).manual_seed(seed + 2000)
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        clip.train()
        lr = schedule_lr(epoch, epochs)
        for group in opt.param_groups:
            group["lr"] = lr
        order = rng.permutation(rows)
        total, seen, t0 = 0.0, 0, time.monotonic()
        pairs: dict[str, int] = {}
        for s in range(0, len(order) - SSL_BATCH + 1, SSL_BATCH):
            r = order[s : s + SSL_BATCH]
            b = source.batch(r, device)
            codes = torch.randint(8, (len(r),), generator=aug, device=device)
            b["image"] = d4(b["image"], codes)
            opt.zero_grad(set_to_none=True)
            loss, counts = clip(b)
            check(torch.isfinite(loss), "finite CLIP loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP_NORM)
            opt.step()
            total += float(loss.detach()) * len(r)
            seen += len(r)
            for k, n in counts.items():
                pairs[k] = pairs.get(k, 0) + n
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "loss": total / seen,
                "examples": seen,
                "pair_examples": pairs,
                "seconds": time.monotonic() - t0,
            }
        )
        log_event(phase="ssl", seed=seed, **history[-1])
    return clip, history


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
