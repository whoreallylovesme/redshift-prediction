from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from lib.train.lightning import fit_best
from lib.utils.utils import pick_device


class LikelihoodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: int = 17
    epochs: int = Field(default=20, gt=0)
    batch_size: int = Field(default=128, gt=0)
    lr: float = Field(default=0.001, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    device: str = "cpu"

    def training_device(self) -> torch.device:
        return pick_device(self.device)


def row_loader(
    dataset: Dataset, rows: NDArray[np.int64], recipe: LikelihoodConfig, *, shuffle: bool = False
) -> DataLoader:
    return DataLoader(
        Subset(dataset, rows.tolist()),
        batch_size=recipe.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(recipe.seed),
    )


class LikelihoodModule(L.LightningModule):
    def __init__(self, model: torch.nn.Module, recipe: LikelihoodConfig) -> None:
        super().__init__()
        self.model = model
        self.recipe = recipe
        self.history: list[dict[str, float]] = []

    def configure_optimizers(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.recipe.lr, weight_decay=self.recipe.weight_decay
        )

    def nll(self, batch: list[Tensor], *, training: bool) -> Tensor:
        raise NotImplementedError

    def _log_nll(self, name: str, value: Tensor, size: int) -> None:
        value = value.detach()
        if value.device.type != "mps":
            value = value.double()
        self.log(name, value, on_epoch=True, on_step=False, batch_size=size)

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        loss = self.nll(batch, training=True)
        self._log_nll("train/nll", loss, len(batch[0]))
        return loss

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> None:
        self._log_nll("val/nll", self.nll(batch, training=False), len(batch[0]))

    def on_train_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        train, val = float(metrics["train/nll"]), float(metrics["val/nll"])
        if not (math.isfinite(train) and math.isfinite(val)):
            raise ValueError("Nonfinite training or validation likelihood")
        self.history.append({"epoch": self.current_epoch + 1, "train/nll": train, "val/nll": val})

    def fit_selected(
        self, out: Path, train: DataLoader, valid: DataLoader, metadata: dict[str, Any]
    ) -> None:
        device = self.recipe.training_device()
        out.mkdir(parents=True, exist_ok=False)
        epoch, score = fit_best(
            self, train, valid, device, self.recipe.epochs, monitor="val/nll",
            gradient_clip=1.0, export_dir=out,
        )
        self.to(device).eval()
        report = {
            **metadata,
            "selected_epoch": epoch,
            "validation_nll": score,
            "history": self.history,
        }
        (out / "train_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
