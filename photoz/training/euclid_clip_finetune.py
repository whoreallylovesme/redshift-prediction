from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import lightning as L
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from lib.train.lightning import predict_batches, single_device_trainer
from lib.utils.log import log_event
from lib.utils.torch_utils import configure_torch
from lib.utils.utils import check, load_yaml_as, pick_device
from photoz.data.euclid_packet import (
    CLIP_NORM,
    FINE_BATCH,
    FINE_EPOCHS,
    INFER_BATCH,
    IT_ARGS,
    LR,
    TABLE_SLOTS,
    WD,
    Packet,
    d4,
    load_it_from_clip,
    masked,
    reset_rng,
    state_copy,
)
from photoz.metrics import bias, delta, nmad, outlier_fraction
from photoz.models.euclid_clip import BinNet, ITNetwork, bin_summary, soft_target
from photoz.training.clip_epochs import (
    ClipTrainingEpoch,
    OptimizerSchedule,
    ShuffledRowBatches,
    scheduled_adamw,
)

TRAIN_KEYS = ("image", "image_available", "table_values", "table_mask", "table_available")
TABLE_KEYS = ("table_values", "table_mask", "table_available")
DEEP_SETS = ("cdfs", "muse", "hos")
HO_FIELDS = {"muse": 1, "hos": 2}
VALID_REFERENCE = "valid_original_reference"
RESERVED_SETS = ("valid", "north", "zval")
PRED_COLUMNS = ("base", "mu", "s")
ZVAL_SIZE = 1500
ZVAL_SEED = 20261106
ZVAL_BINS = 12
ZVAL_Z_MAX = 3.0
ZVAL_BIN_CAP = 0.15
Z_REWEIGHT = 0.5
REWEIGHT_BINS = 30
REWEIGHT_MAX = 10.0
EVAL_EVERY = 5
RECIPE = {"head": "free", "bins": True, "bands": "all"}

type Arrays = dict[str, NDArray[Any]]
type TableRow = tuple[NDArray[np.float32], NDArray[np.bool_], bool]
type BatchFn = Callable[[int, int], dict[str, Tensor]]


class EvalSet(BaseModel):
    name: str
    packet: Path
    phz: Path | None = None


class InferenceConfig(BaseModel):
    out_dir: Path
    seeds: list[int] = Field(default_factory=list)
    state_dir: Path | None = None


class EuclidClipConfig(BaseModel):
    seed: int
    tag: str = "P_final"
    clip_state: Path
    out_dir: Path
    packet: Path
    packet_phz: Path | None = None
    cdfs_packet: Path
    cdfs_labels: Path
    ho_packet: Path
    ho_photometry: Path
    ho_reference: Path
    table_override: list[Path]
    eval_sets: list[EvalSet] = Field(default_factory=list)
    epochs: int = FINE_EPOCHS
    z_reweight: float = Z_REWEIGHT
    zval_size: int = ZVAL_SIZE
    zval_seed: int = ZVAL_SEED
    device: str = "auto"
    num_threads: int = 4
    memory_fraction: float = 0.2
    inference: InferenceConfig | None = None


@dataclass
class LabelledPool:
    ids: NDArray[np.int64]
    z: NDArray[np.float64]
    sources: list[Path]
    source: NDArray[np.int64]
    row: NDArray[np.int64]
    n_fit: int
    deep_counts: dict[str, int]


@dataclass
class ZvalSplit:
    pool: LabelledPool
    overrides: dict[int, TableRow]
    mask: NDArray[np.bool_]
    per_bin: list[int]

    @property
    def ids(self) -> NDArray[np.int64]:
        return self.pool.ids[self.mask]

    @property
    def z(self) -> NDArray[np.float64]:
        return self.pool.z[self.mask]


@dataclass
class SetPrediction:
    object_id: NDArray[np.int64]
    base: NDArray[np.float64]
    mu: NDArray[np.float64]
    s: NDArray[np.float64]
    phz_ok: NDArray[np.bool_]

    @property
    def z(self) -> NDArray[np.float64]:
        out: NDArray[np.float64] = np.expm1(self.base + self.mu)
        return out


def load_config(config_path: Path) -> EuclidClipConfig:
    cfg = load_yaml_as(str(config_path), EuclidClipConfig)
    names = [s.name for s in cfg.eval_sets]
    check(len(set(names)) == len(names), "evaluation set names are unique")
    check(not set(names) & set(RESERVED_SETS), f"evaluation set names avoid {RESERVED_SETS}")
    return cfg


def photoz_metrics(z_true: NDArray[np.float64], z_pred: NDArray[np.float64]) -> dict[str, Any]:
    truth = pd.Series(z_true)
    check(np.isfinite(delta(truth, z_pred)).all(), "finite predictions")
    return {
        "n": int(len(z_true)),
        "nmad": nmad(truth, z_pred),
        "bias": bias(truth, z_pred),
        "outliers": outlier_fraction(truth, z_pred),
    }


def read_rows(dataset: h5py.Dataset, rows: NDArray[np.int64]) -> NDArray[Any]:
    out: NDArray[Any] = np.empty((len(rows),) + tuple(dataset.shape[1:]), dataset.dtype)
    if not len(rows):
        return out
    order = np.argsort(rows, kind="stable")
    ordered = rows[order]
    breaks = np.flatnonzero(np.diff(ordered) != 1) + 1
    for s, e in zip(np.r_[0, breaks], np.r_[breaks, len(ordered)], strict=True):
        out[order[s:e]] = dataset[int(ordered[s]) : int(ordered[e - 1]) + 1]
    return out


def load_overrides(paths: Sequence[Path]) -> dict[int, TableRow]:
    out: dict[int, TableRow] = {}
    for path in paths:
        with np.load(path) as d:
            ids = d["object_id"]
            values, mask, available = d["table_values"], d["table_mask"], d["table_available"]
            status = d["status"] if "status" in d.files else np.zeros(len(ids), np.uint8)
        for j, o in enumerate(ids):
            if int(status[j]) == 0:
                out[int(o)] = (values[j], mask[j], bool(available[j]))
    return out


def apply_overrides(
    arrays: Arrays, ids: NDArray[np.int64], overrides: Mapping[int, TableRow]
) -> int:
    check(set(TABLE_KEYS) <= set(arrays), "table override needs all table arrays")
    hit = 0
    for j, o in enumerate(ids):
        v = overrides.get(int(o))
        if v is not None:
            arrays["table_values"][j], arrays["table_mask"][j], arrays["table_available"][j] = v
            hit += 1
    return hit


def cdfs_labels(
    packet: Path, labels: Path
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    with h5py.File(packet, "r") as h:
        ids, tiles = h["object_id"][:], h["owner_tile"][:]
    lab = pd.read_parquet(labels).drop_duplicates("object_id").set_index("object_id")
    keep = np.isin(ids, lab.index.values)
    z = np.full(len(ids), np.nan)
    z[keep] = lab.loc[ids[keep]]["z"].to_numpy(np.float64)
    return ids, tiles, z, keep


def ho_labels(
    packet: Path, photometry: Path, reference: Path, field_code: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    with h5py.File(packet, "r") as h:
        ids, tiles = h["object_id"][:], h["owner_tile"][:]
    with np.load(photometry, allow_pickle=False) as nat:
        nat_ids, nat_fields = nat["object_id"], nat["field_code"]
    pos = {int(o): i for i, o in enumerate(nat_ids)}
    field = nat_fields[np.array([pos[int(o)] for o in ids], np.int64)]
    ref = pd.read_parquet(reference).drop_duplicates("object_id").set_index("object_id").loc[ids]
    z = ref["z_reference"].to_numpy(float)
    ok = (
        (ref["reference_status"].to_numpy() == VALID_REFERENCE)
        & np.isfinite(z)
        & ~ref["reference_outside_model_grid"].to_numpy(bool)
    )
    return ids, tiles, z, ok & (field == field_code)


def labelled_pool(
    packet: Packet,
    cdfs_packet: Path,
    cdfs_table: Path,
    ho_packet: Path,
    ho_photometry: Path,
    ho_reference: Path,
) -> LabelledPool:
    ti, vi = packet.indices["train"], packet.indices["valid"]
    owner = packet.f["owner_tile"][:]
    fvalid_tiles = np.unique(owner[vi])
    known = {int(o) for o in packet.ids[np.r_[ti, vi]]}
    sources = [packet.path]
    ids = [packet.ids[ti]]
    z = [packet.z["train"].astype(np.float64)]
    source = [np.zeros(len(ti), np.int64)]
    row = [ti.astype(np.int64)]
    deep_counts: dict[str, int] = {}
    for name in DEEP_SETS:
        if name == "cdfs":
            path = cdfs_packet
            set_ids, tiles, zz, keep = cdfs_labels(cdfs_packet, cdfs_table)
        else:
            path = ho_packet
            set_ids, tiles, zz, keep = ho_labels(
                ho_packet, ho_photometry, ho_reference, HO_FIELDS[name]
            )
        keep = (
            keep
            & np.isfinite(zz)
            & (zz >= 0)
            & ~np.isin(tiles, fvalid_tiles)
            & ~np.isin(set_ids, list(known))
        )
        if path not in sources:
            sources.append(path)
        ids.append(set_ids[keep])
        z.append(zz[keep])
        source.append(np.full(int(keep.sum()), sources.index(path), np.int64))
        row.append(np.flatnonzero(keep).astype(np.int64))
        known |= {int(o) for o in set_ids[keep]}
        deep_counts[name] = int(keep.sum())
    n_fit = sum(len(x) for x in ids)
    ids.append(packet.ids[vi])
    z.append(packet.z["valid"].astype(np.float64))
    source.append(np.zeros(len(vi), np.int64))
    row.append(vi.astype(np.int64))
    return LabelledPool(
        ids=np.concatenate(ids).astype(np.int64),
        z=np.concatenate(z),
        sources=sources,
        source=np.concatenate(source),
        row=np.concatenate(row),
        n_fit=n_fit,
        deep_counts=deep_counts,
    )


def pool_arrays(
    pool: LabelledPool,
    rows: NDArray[np.int64],
    overrides: Mapping[int, TableRow],
    keys: Sequence[str] = TRAIN_KEYS,
) -> Arrays:
    out: Arrays = {}
    src, loc = pool.source[rows], pool.row[rows]
    for i, path in enumerate(pool.sources):
        m = np.flatnonzero(src == i)
        with h5py.File(path, "r") as h:
            for k in keys:
                ds = h[k]
                if k not in out:
                    out[k] = np.empty((len(rows),) + tuple(ds.shape[1:]), ds.dtype)
                check(out[k].dtype == ds.dtype, f"{k} dtype agrees across packets")
                if len(m):
                    out[k][m] = read_rows(ds, loc[m])
    if set(TABLE_KEYS) <= set(out):
        apply_overrides(out, pool.ids[rows], overrides)
    return out


def zval_split(
    z: NDArray[np.float64], size: int = ZVAL_SIZE, seed: int = ZVAL_SEED
) -> tuple[NDArray[np.bool_], list[int]]:
    ry, rr = np.log1p(z), np.random.default_rng(seed)
    edges = np.linspace(0.0, np.log1p(ZVAL_Z_MAX), ZVAL_BINS + 1)
    bins = (
        np.flatnonzero((ry >= lo) & (ry < hi)) for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    )
    sel = [
        rr.choice(ix, min(size // ZVAL_BINS, int(ZVAL_BIN_CAP * len(ix))), replace=False)
        for ix in bins
    ]
    mask = np.zeros(len(z), bool)
    mask[np.concatenate(sel)] = True
    return mask, [int(len(x)) for x in sel]


def prepare_split(cfg: EuclidClipConfig, packet: Packet) -> ZvalSplit:
    overrides = load_overrides(cfg.table_override)
    pool = labelled_pool(
        packet,
        cfg.cdfs_packet,
        cfg.cdfs_labels,
        cfg.ho_packet,
        cfg.ho_photometry,
        cfg.ho_reference,
    )
    fit_ids = pool.ids[: pool.n_fit]
    log_event(
        phase="table_override",
        rows=int(len(fit_ids)),
        replaced=sum(int(o) in overrides for o in fit_ids),
    )
    mask, per_bin = zval_split(pool.z, cfg.zval_size, cfg.zval_seed)
    if not mask.any():
        raise ValueError("ZVAL has no rows under the configured stratified sampling policy")
    return ZvalSplit(pool=pool, overrides=overrides, mask=mask, per_bin=per_bin)


def split_counts(split: ZvalSplit) -> dict[str, Any]:
    valid_part = split.mask[split.pool.n_fit :]
    return {
        **split.pool.deep_counts,
        "zval": {
            "n": int(split.mask.sum()),
            "from_fvalid": int(valid_part.sum()),
            "fvalid_into_train": int((~valid_part).sum()),
            "per_bin": split.per_bin,
        },
    }


def z_weights(z: NDArray[np.float64], power: float) -> NDArray[np.float64]:
    if power <= 0:
        return np.ones(len(z))
    lz = np.log1p(z)
    edges = np.linspace(0, lz.max() + 1e-9, REWEIGHT_BINS + 1)
    cnt = np.histogram(lz, edges)[0].astype(float)
    idx = np.clip(np.digitize(lz, edges) - 1, 0, REWEIGHT_BINS - 1)
    wt = np.maximum(cnt[idx], 1.0) ** (-power)
    wt = np.minimum(wt / wt.mean(), REWEIGHT_MAX)
    out: NDArray[np.float64] = wt / wt.mean()
    return out


def phz_available(ids: NDArray[np.int64], lookup: Path | None) -> NDArray[np.bool_]:
    if lookup is None:
        return np.zeros(len(ids), bool)
    with np.load(lookup) as ph:
        object_ids = ph["object_id"]
        median, mode = ph["PHZ_MEDIAN"].reshape(-1), ph["PHZ_MODE_1"].reshape(-1)
        interval = ph["PHZ_70_INT"].reshape(len(object_ids), -1)
    pos = {int(o): i for i, o in enumerate(object_ids)}
    k = np.array([pos.get(int(o), -1) for o in ids], np.int64)
    kk = np.maximum(k, 0)
    med = np.where(k >= 0, median[kk], np.nan).astype(np.float64)
    mod = np.where(k >= 0, mode[kk], np.nan).astype(np.float64)
    width = np.where(k >= 0, interval[kk, -1] - interval[kk, 0], np.nan).astype(np.float64)
    ok: NDArray[np.bool_] = (
        (k >= 0) & np.isfinite(med) & (med >= 0) & np.isfinite(mod) & np.isfinite(width)
    )
    return ok


def band_keep(device: torch.device) -> Tensor:
    return torch.as_tensor(np.ones(TABLE_SLOTS, bool), device=device)


def array_batches(
    data: Mapping[str, NDArray[Any]], descriptor: Tensor, device: torch.device
) -> BatchFn:
    def get(start: int, end: int) -> dict[str, Tensor]:
        b = {k: torch.from_numpy(v[start:end]).to(device) for k, v in data.items()}
        b["table_descriptor"] = descriptor
        return b

    return get


def h5_batches(
    handle: h5py.File,
    rows: NDArray[np.int64],
    ids: NDArray[np.int64],
    overrides: Mapping[int, TableRow],
    descriptor: Tensor,
    device: torch.device,
) -> BatchFn:
    def get(start: int, end: int) -> dict[str, Tensor]:
        part = {k: read_rows(handle[k], rows[start:end]) for k in TRAIN_KEYS}
        if overrides:
            apply_overrides(part, ids[start:end], overrides)
        return array_batches(part, descriptor, device)(0, end - start)

    return get


def index_ranges(n: int) -> DataLoader:
    starts = torch.arange(0, n, INFER_BATCH)
    ranges = TensorDataset(starts, torch.clamp(starts + INFER_BATCH, max=n))
    return DataLoader(ranges, batch_size=None, generator=torch.Generator())


def bin_statistics(
    model: BinNet, get: BatchFn, batch: tuple[Tensor, Tensor], keep: Tensor
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    start, end = batch
    data = masked(get(int(start), int(end)), keep)
    logits: Tensor = model(**{k: data[k] for k in IT_ARGS})
    return bin_summary(torch.softmax(logits.float(), 1).cpu().numpy())


class BinPredictor(L.LightningModule):
    def __init__(self, model: BinNet, get: BatchFn, keep: Tensor) -> None:
        super().__init__()
        self.model = model
        self.get = get
        self.keep = keep

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        median, width = bin_statistics(self.model, self.get, batch, self.keep)
        return torch.from_numpy(np.column_stack((median, width)))


def predict(
    model: BinNet, get: BatchFn, n: int, keep: Tensor
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    model.eval()
    batches = predict_batches(BinPredictor(model, get, keep), index_ranges(n), keep.device)
    values = torch.cat(batches).cpu().numpy() if batches else np.empty((0, 2), dtype=np.float64)
    return values[:, 0], values[:, 1]


def set_prediction(
    model: BinNet,
    get: BatchFn,
    ids: NDArray[np.int64],
    phz_ok: NDArray[np.bool_],
    keep: Tensor,
) -> SetPrediction:
    mu, s = predict(model, get, len(ids), keep)
    return SetPrediction(object_id=ids, base=np.zeros(len(ids)), mu=mu, s=s, phz_ok=phz_ok)


def predict_sets(
    model: BinNet,
    cfg: EuclidClipConfig,
    packet: Packet,
    split: ZvalSplit,
    zval: Mapping[str, NDArray[Any]],
    device: torch.device,
) -> dict[str, SetPrediction]:
    keep = band_keep(device)
    descriptor = torch.as_tensor(packet.descriptor).to(device)
    out: dict[str, SetPrediction] = {}
    for name, rows in (("valid", packet.indices["valid"]), ("north", packet.indices["test"])):
        ids = packet.ids[rows]
        get = h5_batches(packet.f, rows, ids, split.overrides, descriptor, device)
        out[name] = set_prediction(model, get, ids, phz_available(ids, cfg.packet_phz), keep)
    for spec in cfg.eval_sets:
        with h5py.File(spec.packet, "r") as h:
            ids = h["object_id"][:]
            rows = np.arange(len(ids), dtype=np.int64)
            get = h5_batches(h, rows, ids, {}, descriptor, device)
            phz_ok = phz_available(ids, spec.phz)
            out[spec.name] = set_prediction(model, get, ids, phz_ok, keep)
    zval_ids = split.ids
    get = array_batches(zval, descriptor, device)
    out["zval"] = set_prediction(model, get, zval_ids, np.zeros(len(zval_ids), bool), keep)
    return out


def save_predictions(path: Path, preds: Mapping[str, SetPrediction]) -> None:
    flat: dict[str, Any] = {
        f"{name}_{c}": getattr(p, c) for name, p in preds.items() for c in PRED_COLUMNS
    }
    flat.update({f"{name}_object_id": p.object_id for name, p in preds.items()})
    flat.update({f"{name}_phz_ok": p.phz_ok for name, p in preds.items()})
    np.savez(path, **flat)


def build_model(clip_state: Path, seed: int, device: torch.device) -> BinNet:
    it = load_it_from_clip(clip_state)
    reset_rng(seed + 9000)
    return BinNet(it).to(device)


def load_finetuned(path: Path, device: torch.device) -> BinNet:
    model = BinNet(ITNetwork())
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return model.to(device)


class BinnedFineTuning(ClipTrainingEpoch):
    def __init__(
        self,
        model: BinNet,
        train: Mapping[str, NDArray[Any]],
        z: NDArray[np.float64],
        zval: Mapping[str, NDArray[Any]],
        z_val: NDArray[np.float64],
        descriptor: Tensor,
        cfg: EuclidClipConfig,
        device: torch.device,
    ) -> None:
        super().__init__(cfg.epochs)
        self.model = model
        self.train_arrays = train
        self.cfg = cfg
        self.gen = torch.Generator(device=device).manual_seed(cfg.seed + 4000)
        self.descriptor = descriptor
        self.keep = band_keep(device)
        self.validation_medians: list[NDArray[np.float64]] = []
        self.zval = zval
        self.z_val = z_val
        wt = z_weights(z, cfg.z_reweight)
        if cfg.z_reweight > 0:
            log_event(
                phase="z_reweight", power=cfg.z_reweight, w_min=float(wt.min()),
                w_max=float(wt.max()),
            )
        self.weights = torch.as_tensor(wt.astype("f4"))
        self.target = torch.as_tensor(soft_target(np.log1p(z)))

    def configure_optimizers(self) -> OptimizerSchedule:
        return scheduled_adamw(self.model.parameters(), self.epochs, lr=LR, weight_decay=WD)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loc = batch.numpy()
        data = {k: torch.from_numpy(self.train_arrays[k][loc]).to(self.device) for k in TRAIN_KEYS}
        data["table_descriptor"] = self.descriptor
        codes = torch.randint(8, (len(loc),), generator=self.gen, device=self.device)
        data["image"] = d4(data["image"], codes)
        data = masked(data, self.keep)
        weights = self.weights[loc].to(self.device)
        logits: Tensor = self.model(**{k: data[k] for k in IT_ARGS})
        log_prob = torch.log_softmax(logits.float(), 1)
        loss = (weights * -(self.target[loc].to(self.device) * log_prob).sum(1)).sum()
        loss = loss / weights.sum()
        check(torch.isfinite(loss), "finite loss")
        self.record_loss(loss, len(loc))
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        epoch = self.current_epoch + 1
        if epoch % EVAL_EVERY == 0 or epoch == self.cfg.epochs:
            get = array_batches(self.zval, self.descriptor, self.device)
            median, _ = bin_statistics(self.model, get, batch, self.keep)
            self.validation_medians.append(median)

    def on_train_epoch_end(self) -> None:
        if not self.validation_medians:
            return
        epoch = self.current_epoch + 1
        mu = np.concatenate(self.validation_medians)
        self.validation_medians.clear()
        valid = photoz_metrics(self.z_val, np.expm1(mu))
        self.history.append({
            "epoch": epoch, "lr": self.lr,
            "train_nll": float(self.trainer.callback_metrics["train/loss"]),
            "valid": valid,
        })
        self.log("val/nmad", valid["nmad"])
        log_event(
            phase="finetune", tag=self.cfg.tag, seed=self.cfg.seed, epoch=epoch,
            valid_nmad=round(valid["nmad"], 4), valid_out=round(valid["outliers"], 4),
        )


def fine_tune(
    model: BinNet,
    train: Mapping[str, NDArray[Any]],
    z: NDArray[np.float64],
    zval: Mapping[str, NDArray[Any]],
    z_val: NDArray[np.float64],
    descriptor: Tensor,
    cfg: EuclidClipConfig,
    device: torch.device,
) -> list[dict[str, Any]]:
    reset_rng(cfg.seed + 3000)
    module = BinnedFineTuning(model, train, z, zval, z_val, descriptor, cfg, device)
    batches = ShuffledRowBatches(np.arange(len(z)), cfg.seed, FINE_BATCH, drop_last=False)
    trainer = single_device_trainer(device, cfg.epochs, gradient_clip=CLIP_NORM)
    trainer.fit(module, batches.loader(), index_ranges(len(z_val)))
    model.to(device)
    return module.history


def run_train(config_path: Path) -> None:
    cfg = load_config(config_path)
    name = f"{cfg.tag}_{cfg.seed}"
    meta_path = cfg.out_dir / f"{name}.json"
    check(not meta_path.exists(), f"exclusive output: {meta_path}")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    configure_torch(cfg.num_threads, cfg.memory_fraction, device)
    start = time.monotonic()
    packet = Packet(cfg.packet)
    split = prepare_split(cfg, packet)
    counts = split_counts(split)
    train_rows = np.flatnonzero(~split.mask)
    train = pool_arrays(split.pool, train_rows, split.overrides)
    zval = pool_arrays(split.pool, np.flatnonzero(split.mask), split.overrides)
    z, z_val = split.pool.z[train_rows], split.z
    log_event(
        phase="train_rows",
        f_train=int(len(packet.indices["train"])),
        deep=counts,
        total=int(len(z)),
    )
    model = build_model(cfg.clip_state, cfg.seed, device)
    descriptor = torch.as_tensor(packet.descriptor).to(device)
    history = fine_tune(model, train, z, zval, z_val, descriptor, cfg, device)
    keep = band_keep(device)
    _, train_s = predict(model, array_batches(train, descriptor, device), len(z), keep)
    s90, s99 = float(np.percentile(train_s, 90)), float(np.percentile(train_s, 99))
    state_path = cfg.out_dir / f"{name}_state.pt"
    torch.save(state_copy(model), state_path)
    preds = predict_sets(model, cfg, packet, split, zval, device)
    prediction = cfg.out_dir / f"{name}_pred.npz"
    save_predictions(prediction, preds)
    prediction.with_suffix(".history.json").write_text(json.dumps(history, indent=1))
    z_fvalid = packet.z["valid"].astype(np.float64)
    meta = {
        "args": {**RECIPE, **cfg.model_dump(mode="json")},
        "deep_counts": counts,
        "n_train": int(len(z)),
        "s90": s90,
        "s99": s99,
        "zval_metrics": history[-1]["valid"],
        "metric_partitions": {"fvalid_ungated": "zval", "fvalid_gated": "original_F_VALID"},
        "fvalid_ungated": history[-1]["valid"],
        "fvalid_gated": photoz_metrics(z_fvalid, preds["valid"].z),
        "history": history,
        "zval": {
            "ids": [int(o) for o in split.ids],
            "z": [float(x) for x in z_val],
        },
        "seconds": time.monotonic() - start,
    }
    meta_path.write_text(json.dumps(meta, indent=1) + "\n")
    log_event(
        phase="finetune_done",
        tag=cfg.tag,
        seed=cfg.seed,
        fvalid=history[-1]["valid"],
        seconds=time.monotonic() - start,
    )
    packet.close()
