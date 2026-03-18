from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class GalaxyDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        datasets: tuple[str, ...] = ("teddy", "happy"),
        splits: tuple[str, ...] = ("A", "B", "C", "D"),
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        augment: bool = False,
    ):
        self.root = Path(root)
        self.samples: list[tuple[Path, int]] = []
        for label, ds in enumerate(datasets):
            for split in splits:
                d = self.root / ds / split
                if not d.exists():
                    continue
                for p in d.glob("*.npz"):
                    self.samples.append((p, label))
        self.mean = mean if mean is not None else np.zeros(5, dtype=np.float32)
        self.std = std if std is not None else np.ones(5, dtype=np.float32)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        data = np.load(path)
        img = data["image"].astype(np.float32)
        img = (img - self.mean) / self.std
        if self.augment:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=0).copy()
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1).copy()
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k).copy()
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img, label


def get_dataloaders(
    root: str | Path,
    batch_size: int = 32,
    num_workers: int = 0,
    train_ratio: float = 0.8,
    augment: bool = True,
    seed: int = 42,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    full = GalaxyDataset(root)
    if len(full) == 0:
        raise ValueError(f"No .npz files found in {root}")

    labels = [label for _, label in full.samples]
    indices = list(range(len(full)))
    counts = Counter(labels)
    stratify = labels if min(counts.values()) >= 2 else None
    train_idx, val_idx = train_test_split(
        indices, train_size=train_ratio, stratify=stratify, random_state=seed
    )

    train_ds = GalaxyDataset(root, augment=augment)
    train_ds.samples = [full.samples[i] for i in train_idx]
    val_ds = GalaxyDataset(root, augment=False)
    val_ds.samples = [full.samples[i] for i in val_idx]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
