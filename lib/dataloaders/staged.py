from dataclasses import dataclass
from typing import Any

from torch.utils.data import DataLoader


@dataclass(frozen=True)
class StagedLoaders:
    train_a: DataLoader[Any] | None
    val_a: DataLoader[Any] | None
    train_b: DataLoader[Any]
    val_b: DataLoader[Any]
