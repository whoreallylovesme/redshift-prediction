from __future__ import annotations

from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lib.dataloaders.common import (
    center_crop,
    inverse_bin_weights,
    random_rot90_flip,
    subsample_indices,
)
from lib.dataloaders.hdf5_rows import HDF5Rows
from lib.utils.log import get_logger

logger = get_logger()


class HayatDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):

    def __init__(
        self,
        h5_path: Path,
        indices: np.ndarray,
        z_labels: np.ndarray,
        morph_labels: np.ndarray,
        image_scale: float = 10.0,
        crop_size: int = 64,
        augment: bool = False,
        rot90_prob: float = 0.67,
        flip_prob: float = 0.33,
    ) -> None:
        self.h5_path = h5_path
        self.indices = indices.astype(np.int64)
        self.z_labels = z_labels.astype(np.float32)
        self.morph_labels = morph_labels.astype(np.float32)
        self.image_scale = float(image_scale)
        self.crop_size = crop_size
        self.augment = augment
        self.rot90_prob = float(rot90_prob)
        self.flip_prob = float(flip_prob)

    @cached_property
    def images(self) -> HDF5Rows:
        return HDF5Rows(self.h5_path, "images", swmr=True)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.images[int(self.indices[idx])]
        img = np.transpose(img, (1, 2, 0))

        if self.crop_size < img.shape[0]:
            img = center_crop(img, self.crop_size)

        img = img.astype(np.float32) / self.image_scale

        if self.augment:
            img = random_rot90_flip(img, self.rot90_prob, self.flip_prob)

        tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        z = torch.tensor(self.z_labels[idx], dtype=torch.float32)
        m = torch.tensor(self.morph_labels[idx], dtype=torch.float32)
        return tensor, z, m


def _worker_init_fn(worker_id: int) -> None:
    np.random.seed((torch.initial_seed() + worker_id) % (2**31 - 1))


def get_hayat_dataloaders(
    h5_path: str | Path,
    morph_parquet: str | Path,
    batch_size: int = 400,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42,
    max_samples: int | None = None,
    weighted_sampling: bool = True,
    crop_size: int = 64,
    image_scale: float = 10.0,
    labeled_only: bool = False,
    rot90_prob: float = 0.67,
    flip_prob: float = 0.33,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    h5_path = Path(h5_path)
    morph_df = pd.read_parquet(morph_parquet)

    with h5py.File(h5_path, "r") as f:
        n = f["images"].shape[0]
        z_all = f["specz_redshift"][:].astype(np.float32)

    if len(morph_df) != n:
        raise ValueError("Morphology table must have one row per HDF5 image")

    indices = np.arange(n)
    morph_all = morph_df["gz_morpho_binary"].to_numpy(dtype=np.float32)

    if max_samples is not None and n > max_samples:
        keep = subsample_indices(n, int(max_samples), seed)
        indices = indices[keep]
        z_all = z_all[keep]
        morph_all = morph_all[keep]

    train_pos, val_pos = train_test_split(
        np.arange(len(indices)), train_size=train_ratio, random_state=seed
    )
    if labeled_only:
        train_pos = train_pos[np.isfinite(morph_all[train_pos])]
        val_pos = val_pos[np.isfinite(morph_all[val_pos])]
        if not len(train_pos) or not len(val_pos):
            raise ValueError("CapsNet stage A needs morphology labels in both splits")
    train_idx = indices[train_pos]
    val_idx = indices[val_pos]
    train_z = z_all[train_pos]
    val_z = z_all[val_pos]
    train_m = morph_all[train_pos]
    val_m = morph_all[val_pos]

    logger.info(
        f"[loader] train N={len(train_idx)}, val N={len(val_idx)}"
        f"{' (labeled_only)' if labeled_only else ''}",
    )
    logger.info(
        f"[loader] train morph labeled: {int((~np.isnan(train_m)).sum())} "
        f"(spiral={int((train_m==1).sum())}, ellip={int((train_m==0).sum())})",
    )
    logger.info(f"[loader] image_scale={image_scale}, crop_size={crop_size}")

    train_ds = HayatDataset(
        h5_path, train_idx, train_z, train_m,
        image_scale=image_scale, crop_size=crop_size, augment=True,
        rot90_prob=rot90_prob, flip_prob=flip_prob,
    )
    val_ds = HayatDataset(
        h5_path, val_idx, val_z, val_m,
        image_scale=image_scale, crop_size=crop_size, augment=False,
    )

    sampler: WeightedRandomSampler | None = None
    shuffle = True
    if weighted_sampling:
        weights = inverse_bin_weights(train_z)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double().tolist(),
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
        drop_last=len(train_ds) >= batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
