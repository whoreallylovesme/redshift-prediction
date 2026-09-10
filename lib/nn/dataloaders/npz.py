from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lib.nn.dataloaders.common import subsample_indices


def _center_crop_resize(img: np.ndarray, crop_size: int, out_size: int = 224) -> np.ndarray:
    h, w = img.shape[:2]
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    img = img[top : top + crop_size, left : left + crop_size]
    if crop_size != out_size:
        t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
        img = t.squeeze(0).permute(1, 2, 0).numpy()
    return img

def parse_catalog(catalog_path: str | Path) -> dict[str, float]:
    mapping: dict[str, float] = {}
    with open(catalog_path) as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.split()
            if len(parts) >= 7:
                objid = parts[0]
                z_spec = float(parts[6])
                mapping[objid] = z_spec
    return mapping

def _resolve_w1w2_path(griz_path: Path, w1w2_dir: Path) -> Path:
    return w1w2_dir / f"{griz_path.stem}_wise.npz"

class PhotozDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        files: list[Path],
        labels: list[float],
        mean: npt.NDArray[np.floating] | None = None,
        std: npt.NDArray[np.floating] | None = None,
        augment: bool = False,
        w1w2_dir: Path | None = None,
        in_channels: int = 4,
        crop_size: int | None = None,
        out_size: int = 224,
        physics_augment: bool = False,
    ) -> None:
        self.files = files
        self.labels = labels
        self.w1w2_dir = w1w2_dir
        self.in_channels = in_channels
        self.mean = mean if mean is not None else np.zeros(in_channels, dtype=np.float32)
        self.std = std if std is not None else np.ones(in_channels, dtype=np.float32)
        self.augment = augment
        self.crop_size = crop_size
        self.out_size = out_size
        self.physics_augment = physics_augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.load(self.files[idx])["image"].astype(np.float32)[:, :, : self.in_channels]

        if self.w1w2_dir is not None:
            w1w2_path = _resolve_w1w2_path(self.files[idx], self.w1w2_dir)
            w1w2 = np.load(w1w2_path)["image"].astype(np.float32)[:, :, :2]
            img = np.concatenate([img, w1w2], axis=2)

        label_val = self.labels[idx]

        if self.augment and self.physics_augment and np.random.rand() < 0.4:
            delta_z = float(np.random.uniform(0.0, 0.5))
            z_aug = label_val + delta_z
            flux_factor = float(((1.0 + label_val) / (1.0 + z_aug)) ** 4)
            img = img * flux_factor
            label_val = z_aug

        if self.crop_size is not None and self.crop_size < img.shape[0]:
            img = _center_crop_resize(img, self.crop_size, out_size=self.out_size)

        img = (img - self.mean) / (self.std + 1e-8)

        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=0).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k).copy()

        tensor = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.tensor(label_val, dtype=torch.float32)
        return tensor, label

def compute_stats(
    files: list[Path],
    n_samples: int = 2000,
    w1w2_dir: Path | None = None,
    in_channels: int = 4,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    rng = np.random.default_rng(42)
    subset = rng.choice(len(files), size=min(n_samples, len(files)), replace=False)
    acc = []
    for i in subset:
        img = np.load(files[i])["image"].astype(np.float32)[:, :, :in_channels]
        if w1w2_dir is not None:
            w1w2_path = _resolve_w1w2_path(files[i], w1w2_dir)
            w1w2 = np.load(w1w2_path)["image"].astype(np.float32)[:, :, :2]
            img = np.concatenate([img, w1w2], axis=2)
        acc.append(img.reshape(-1, img.shape[-1]))
    all_pixels: npt.NDArray[np.floating] = np.concatenate(acc, axis=0)
    return all_pixels.mean(axis=0).astype(np.float32), all_pixels.std(axis=0).astype(np.float32)

def _compute_sample_weights(labels: list[float], n_bins: int = 20) -> npt.NDArray[np.float64]:
    arr = np.array(labels)
    bins = np.linspace(arr.min(), arr.max() + 1e-8, n_bins + 1)
    bin_idx = np.digitize(arr, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    counts[counts == 0] = 1.0
    bin_weights = 1.0 / counts
    sample_weights: npt.NDArray[np.float64] = bin_weights[bin_idx]
    return sample_weights

def get_npz_dataloaders(
    data_dir: str | Path,
    catalog_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42,
    max_samples: int | None = None,
    weighted_sampling: bool = False,
    w1w2_dir: str | Path | None = None,
    in_channels: int = 4,
    crop_size: int | None = None,
    out_size: int = 224,
    physics_augment: bool = False,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    data_dir = Path(data_dir)
    _w1w2_dir = Path(w1w2_dir) if w1w2_dir is not None else None
    catalog = parse_catalog(catalog_path)

    files: list[Path] = []
    labels: list[float] = []
    for p in sorted(data_dir.glob("*.npz")):
        objid = p.stem
        if objid not in catalog:
            continue
        if _w1w2_dir is not None and not _resolve_w1w2_path(p, _w1w2_dir).exists():
            continue
        files.append(p)
        labels.append(catalog[objid])

    if len(files) == 0:
        raise ValueError(f"No matching images found in {data_dir}")

    if max_samples is not None and len(files) > max_samples:
        idx = subsample_indices(len(files), int(max_samples), seed)
        files = [files[i] for i in idx]
        labels = [labels[i] for i in idx]

    train_idx, val_idx = train_test_split(
        list(range(len(files))), train_size=train_ratio, random_state=seed
    )

    train_files = [files[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    mean, std = compute_stats(train_files, w1w2_dir=_w1w2_dir, in_channels=in_channels)

    train_ds = PhotozDataset(
        train_files, train_labels, mean, std,
        augment=True, w1w2_dir=_w1w2_dir, in_channels=in_channels,
        crop_size=crop_size, out_size=out_size, physics_augment=physics_augment,
    )
    val_ds = PhotozDataset(
        val_files, val_labels, mean, std,
        augment=False, w1w2_dir=_w1w2_dir, in_channels=in_channels,
        crop_size=crop_size, out_size=out_size,
    )

    sampler = None
    shuffle = True
    if weighted_sampling:
        weights = _compute_sample_weights(train_labels)
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(weights).double().tolist(),
            num_samples=len(train_ds),
            replacement=True,
        )
        shuffle = False

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def load_npz(
    path: Path,
    mean: npt.NDArray[np.floating],
    std: npt.NDArray[np.floating],
    w1w2_path: Path | None = None,
    in_channels: int = 4,
    crop_size: int | None = None,
    out_size: int = 224,
) -> torch.Tensor:
    img = np.load(path)["image"].astype(np.float32)[:, :, :in_channels]
    if w1w2_path is not None:
        w1w2 = np.load(w1w2_path)["image"].astype(np.float32)[:, :, :2]
        img = np.concatenate([img, w1w2], axis=2)
    if crop_size is not None and crop_size < img.shape[0]:
        img = _center_crop_resize(img, crop_size, out_size=out_size)
    img = (img - mean) / (std + 1e-8)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
