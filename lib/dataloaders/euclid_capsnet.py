"""Загрузчик Euclid Q1 EDF-N: штампы VIS/Y/J/H из тайловых npz + метки гризма.

Стыкуется со staged-обучением CapsNet: стадия A греет капсулы на бинарной
морфологии (extended/point-like из MER), стадия B учит редшифт по гризм-меткам.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lib.dataloaders.common import random_rot90_flip
from lib.utils.log import get_logger

logger = get_logger()


class EuclidStampsDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Штампы выбранных объектов; тайлы держим в памяти как float16.

    В labels должны быть колонки object_id, spe_z, extended_prob; берутся
    только те object_id, что найдутся в npz-штампах перечисленных тайлов.
    """

    def __init__(
        self,
        stamps_dir: Path,
        tiles: list[int],
        labels: pd.DataFrame,
        norm_scales: np.ndarray,
        z_min: float,
        z_max: float,
        augment: bool = False,
        rot90_prob: float = 0.67,
        flip_prob: float = 0.33,
    ) -> None:
        label_index = labels.set_index("object_id")
        frames: list[np.ndarray] = []
        rows: list[pd.DataFrame] = []
        for tile in tiles:
            f = stamps_dir / f"{tile}.npz"
            if not f.exists():
                logger.warning(f"[euclid] нет штампов тайла {tile}, пропускаю")
                continue
            with np.load(f) as z:
                ids = z["ids"]
                X = z["X"]
            found = label_index.reindex(ids.tolist())
            mask = found["spe_z"].notna().to_numpy()
            rows.append(found[mask].reset_index())
            frames.append(X[mask])
            logger.info(f"[euclid] тайл {tile}: {int(mask.sum())}/{len(ids)} с метками")
        if not frames:
            raise RuntimeError(f"[euclid] ни одного тайла в {stamps_dir}")
        self.X = np.concatenate(frames, axis=0)
        lab = pd.concat(rows, ignore_index=True)
        self.ids = lab["object_id"].to_numpy(np.int64)
        z_lab = lab["spe_z"].to_numpy(np.float32)
        self.z = np.clip(z_lab, z_min + 1e-5, z_max - 1e-5)
        ext = lab["extended_prob"].fillna(0.5).to_numpy(np.float32)
        self.morph = (ext > 0.5).astype(np.float32)
        self.norm_scales = norm_scales.astype(np.float32)
        self.augment = augment
        self.rot90_prob = rot90_prob
        self.flip_prob = flip_prob

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.X[idx].astype(np.float32)  # [4,H,W]
        img = np.arcsinh(img / self.norm_scales[:, None, None]) / 3.0
        if self.augment:
            img = random_rot90_flip(
                img.transpose(1, 2, 0), self.rot90_prob, self.flip_prob
            ).transpose(2, 0, 1)
        return (
            torch.from_numpy(np.ascontiguousarray(img)),
            torch.tensor(self.z[idx]),
            torch.tensor(self.morph[idx]),
        )


def _norm_scales(
    stamps_dir: Path, tiles: list[int], labels: pd.DataFrame, sample: int = 200_000, seed: int = 0
) -> np.ndarray:
    """Пер-канальные масштабы: p99.5 |потока| на подвыборке, делённый на 5."""
    rng = np.random.default_rng(seed)
    frames = []
    for tile in tiles:
        f = stamps_dir / f"{tile}.npz"
        if f.exists():
            frames.append(np.load(f)["X"])
    X = np.concatenate(frames, axis=0)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    flat = X[idx].astype(np.float32).reshape(len(idx), X.shape[1], -1)
    p99 = np.percentile(np.abs(flat), 99.5, axis=(0, 2))
    return (p99 / 5.0).astype(np.float32)


def _z_balanced_sampler(z: np.ndarray) -> WeightedRandomSampler:
    bins = np.linspace(0.0, 1.0, 9)
    y = (z / (z.max() + 1e-9)).clip(0.0, 0.999)
    b = np.digitize(y, bins)
    counts = np.bincount(b, minlength=9).astype(np.float64)
    w = 1.0 / np.maximum(counts[b], 1.0)
    return WeightedRandomSampler(w.tolist(), len(z), replacement=True)


def _loader(
    ds: Dataset[Any],
    batch_size: int,
    num_workers: int,
    *,
    weighted: bool = False,
    z: np.ndarray | None = None,
) -> DataLoader[Any]:
    sampler = _z_balanced_sampler(z) if weighted and z is not None else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=True,
    )


def get_euclid_capsnet_dataloaders(
    stamps_dir: str,
    labels_parquet: str,
    train_tiles: list[int],
    valid_tiles: list[int],
    stage_a_zprob: float = 0.95,
    save_scales_to: str | None = None,
    batch_size: int = 256,
    num_workers: int = 3,
    z_min: float = 0.0,
    z_max: float = 3.0,
    seed: int = 200,
    rot90_prob: float = 0.67,
    flip_prob: float = 0.33,
    **_: Any,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    labels = pd.read_parquet(labels_parquet)
    stamps = Path(stamps_dir)
    scales = _norm_scales(stamps, train_tiles, labels, seed=seed)
    logger.info(f"[euclid] norm_scales (VIS,Y,J,H): {np.array2string(scales, precision=3)}")
    if save_scales_to:
        Path(save_scales_to).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_scales_to, scales)

    def build(tiles: list[int], lab: pd.DataFrame, augment: bool) -> EuclidStampsDataset:
        return EuclidStampsDataset(
            stamps_dir=stamps,
            tiles=tiles,
            labels=lab,
            norm_scales=scales,
            z_min=z_min,
            z_max=z_max,
            augment=augment,
            rot90_prob=rot90_prob,
            flip_prob=flip_prob,
        )

    train_b = build(train_tiles, labels, augment=True)
    hi = labels[labels["spe_z_prob"] > stage_a_zprob]
    train_a = build(train_tiles, hi, augment=True)
    logger.info(f"[euclid] стадия A: {len(train_a)} объектов с z_prob>{stage_a_zprob}")
    valid = build(valid_tiles, labels, augment=False)

    la = _loader(train_a, batch_size, num_workers)
    va = _loader(valid, batch_size, num_workers)
    lb = _loader(train_b, batch_size, num_workers, weighted=True, z=train_b.z)
    vb = _loader(valid, batch_size, num_workers)
    return la, va, lb, vb
