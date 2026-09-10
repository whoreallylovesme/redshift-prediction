from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lib.log import get_logger
from lib.nn.dataloaders.common import (
    center_crop,
    random_rot90_flip,
    subsample_indices,
    z_sample_weights,
)
from lib.nn.dataloaders.npz import parse_catalog

logger = get_logger()




def _build_morph_map(gz1_parquet: Path) -> dict[str, float]:
    df = pd.read_parquet(gz1_parquet)
    out: dict[str, float] = {}

    for objid, spiral, elliptical in zip(
        df["objid"].astype(str), df["gz_spiral"], df["gz_elliptical"], strict=True,
    ):
        if spiral == 1:
            out[objid] = 1.0
        elif elliptical == 1:
            out[objid] = 0.0
    return out

TARGET_SCALE = 0.05
PER_CHANNEL_NORM_6CH = np.array(
    [
        [0.0003, 0.072],
        [0.0007, 0.243],
        [0.0011, 0.429],
        [0.0012, 0.573],
        [2.0114, 156.717],
        [2.8756, 175.454],
    ],
    dtype=np.float32,
)

class TeddyCapsnetDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        files: list[Path],
        z_labels: np.ndarray,
        morph_labels: np.ndarray,
        in_channels: int = 4,
        crop_size: int = 64,
        image_scale: float = 10.0,
        augment: bool = False,
        per_channel_norm: bool = False,
        w1w2_dir: Path | None = None,
        rot90_prob: float = 0.67,
        flip_prob: float = 0.33,
    ) -> None:
        self.files = files
        self.z_labels = z_labels.astype(np.float32)
        self.morph_labels = morph_labels.astype(np.float32)
        self.in_channels = in_channels
        self.crop_size = crop_size
        self.image_scale = float(image_scale)
        self.augment = augment
        self.per_channel_norm = per_channel_norm and in_channels == 6
        self.w1w2_dir = w1w2_dir
        self.rot90_prob = float(rot90_prob)
        self.flip_prob = float(flip_prob)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        n_griz = self.in_channels - 2 if self.w1w2_dir is not None else self.in_channels
        img = np.load(path)["image"].astype(np.float32)[:, :, :n_griz]
        if self.w1w2_dir is not None:

            wise = np.load(self.w1w2_dir / f"{path.stem}_wise.npz")["image"]
            img = np.concatenate([img, wise.astype(np.float32)[:, :, :2]], axis=2)
        if self.crop_size < img.shape[0]:
            img = center_crop(img, self.crop_size)
        if self.per_channel_norm:
            med = PER_CHANNEL_NORM_6CH[:, 0]
            scale = PER_CHANNEL_NORM_6CH[:, 1] - PER_CHANNEL_NORM_6CH[:, 0]
            img = (img - med) / scale * TARGET_SCALE
        else:
            img = img / self.image_scale

        if self.augment:
            img = random_rot90_flip(img, self.rot90_prob, self.flip_prob)

        tensor = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
        z = torch.tensor(self.z_labels[idx], dtype=torch.float32)
        m = torch.tensor(self.morph_labels[idx], dtype=torch.float32)
        return tensor, z, m


def _scan(
    images_dir: Path, catalog_path: Path, gz1_parquet: Path,
    w1w2_dir: Path | None = None,
) -> tuple[list[Path], np.ndarray, np.ndarray]:
    z_map = parse_catalog(catalog_path)
    morph_map = _build_morph_map(gz1_parquet)

    files: list[Path] = []
    zs: list[float] = []
    ms: list[float] = []
    for npz in sorted(images_dir.iterdir()):
        if npz.suffix != ".npz":
            continue
        objid = npz.stem
        z = z_map.get(objid)
        if z is None:
            continue
        if w1w2_dir is not None and not (w1w2_dir / f"{objid}_wise.npz").exists():
            continue
        files.append(npz)
        zs.append(z)
        ms.append(morph_map.get(objid, float("nan")))
    return files, np.array(zs, dtype=np.float32), np.array(ms, dtype=np.float32)

def get_teddy_capsnet_dataloaders(
    images_dir: str | Path,
    catalog_path: str | Path,
    gz1_parquet: str | Path,
    batch_size: int = 400,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 200,
    max_samples: int | None = None,
    weighted_sampling: bool = True,
    crop_size: int = 64,
    in_channels: int = 4,
    image_scale: float = 10.0,
    labeled_only: bool = False,
    per_channel_norm: bool = False,
    rot90_prob: float = 0.67,
    flip_prob: float = 0.33,
) -> tuple[DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
           DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
    files, z_all, morph_all = _scan(Path(images_dir), Path(catalog_path), Path(gz1_parquet))
    logger.info(
        f"[teddy] scanned {len(files)} npz with z_spec; "
        f"morph labeled: {int((~np.isnan(morph_all)).sum())} "
        f"(spiral={int((morph_all==1).sum())}, ellip={int((morph_all==0).sum())})",
    )

    if labeled_only:
        keep = ~np.isnan(morph_all)
        files = [f for f, k in zip(files, keep, strict=True) if k]
        z_all = z_all[keep]
        morph_all = morph_all[keep]
        logger.info(f"[teddy] labeled_only filter: {len(files)} kept")

    if max_samples is not None and len(files) > max_samples:
        keep_idx = subsample_indices(len(files), int(max_samples), seed)
        files = [files[i] for i in keep_idx]
        z_all = z_all[keep_idx]
        morph_all = morph_all[keep_idx]

    train_pos, val_pos = train_test_split(
        np.arange(len(files)), train_size=train_ratio, random_state=seed,
    )
    train_files = [files[i] for i in train_pos]
    val_files = [files[i] for i in val_pos]
    train_z = z_all[train_pos]
    val_z = z_all[val_pos]
    train_m = morph_all[train_pos]
    val_m = morph_all[val_pos]

    logger.info(
        f"[teddy] train N={len(train_files)}, val N={len(val_files)} "
        f"(train spiral={int((train_m==1).sum())}, ellip={int((train_m==0).sum())}, "
        f"unlabeled={int(np.isnan(train_m).sum())})",
    )

    train_ds = TeddyCapsnetDataset(
        train_files, train_z, train_m,
        in_channels=in_channels, crop_size=crop_size, image_scale=image_scale,
        augment=True, per_channel_norm=per_channel_norm,
        rot90_prob=rot90_prob, flip_prob=flip_prob,
    )
    val_ds = TeddyCapsnetDataset(
        val_files, val_z, val_m,
        in_channels=in_channels, crop_size=crop_size, image_scale=image_scale,
        augment=False, per_channel_norm=per_channel_norm,
    )

    sampler: WeightedRandomSampler | None = None
    shuffle = True
    if weighted_sampling:
        if labeled_only:

            spiral_w = 1.0 / max(int((train_m == 1).sum()), 1)
            ellip_w = 1.0 / max(int((train_m == 0).sum()), 1)
            w = np.where(train_m == 1, spiral_w, ellip_w).astype(np.float64)
        else:
            w = z_sample_weights(train_z)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(w).double().tolist(),
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
