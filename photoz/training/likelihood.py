from pathlib import Path

import numpy as np
import torch
from pydantic import Field

from lib.train.likelihood import LikelihoodConfig
from lib.train.likelihood import LikelihoodModule as LikelihoodModule
from lib.train.likelihood import row_loader as row_loader


class LikelihoodRecipe(LikelihoodConfig):
    projection: Path
    batch_size: int = Field(default=128, gt=0, le=4096)

    def training_device(self) -> torch.device:
        device = super().training_device()
        if device.type not in {"cpu", "cuda"}:
            raise ValueError("The float64 likelihood requires a CPU or CUDA device")
        return device


def partition(ids: np.ndarray, train: np.ndarray, valid: np.ndarray) -> None:
    if ids.dtype != np.int64 or ids.ndim != 1 or len(np.unique(ids)) != len(ids):
        raise ValueError("object_id must contain one unique signed int64 ID per row")
    if any(rows.ndim != 1 or rows.dtype != np.int64 or not len(rows) for rows in (train, valid)):
        raise ValueError("fit_indices and valid_indices must be nonempty int64 vectors")
    if not np.array_equal(np.sort(np.concatenate([train, valid])), np.arange(len(ids))):
        raise ValueError("Training and validation rows must be a disjoint, complete partition")
