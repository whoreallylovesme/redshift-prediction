from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


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


class PhotozDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        files: list[Path],
        labels: list[float],
        mean: npt.NDArray[np.floating] | None = None,
        std: npt.NDArray[np.floating] | None = None,
        augment: bool = False,
    ) -> None:
        self.files = files
        self.labels = labels
        self.mean = mean if mean is not None else np.zeros(4, dtype=np.float32)
        self.std = std if std is not None else np.ones(4, dtype=np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.load(self.files[idx])["image"].astype(np.float32)
        img = img[:, :, :4]

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
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tensor, label


def compute_stats(
    files: list[Path], n_samples: int = 2000
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    rng = np.random.default_rng(42)
    subset = rng.choice(len(files), size=min(n_samples, len(files)), replace=False)
    acc = []
    for i in subset:
        img = np.load(files[i])["image"].astype(np.float32)[:, :, :4]
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


def get_photoz_dataloaders(
    data_dir: str | Path,
    catalog_path: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42,
    max_samples: int | None = None,
    weighted_sampling: bool = False,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    data_dir = Path(data_dir)
    catalog = parse_catalog(catalog_path)

    files: list[Path] = []
    labels: list[float] = []
    for p in sorted(data_dir.glob("*.npz")):
        objid = p.stem
        if objid in catalog:
            files.append(p)
            labels.append(catalog[objid])

    if len(files) == 0:
        raise ValueError(f"No matching images found in {data_dir}")

    if max_samples is not None and len(files) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(files), size=max_samples, replace=False)
        idx.sort()
        files = [files[i] for i in idx]
        labels = [labels[i] for i in idx]


    train_idx, val_idx = train_test_split(
        list(range(len(files))), train_size=train_ratio, random_state=seed
    )

    train_files = [files[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_files = [files[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    mean, std = compute_stats(train_files)

    train_ds = PhotozDataset(train_files, train_labels, mean, std, augment=True)
    val_ds = PhotozDataset(val_files, val_labels, mean, std, augment=False)

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
