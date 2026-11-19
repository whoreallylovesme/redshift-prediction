from __future__ import annotations

import json
import math
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from lib.config import PytorchTrainConfig
from lib.tracking import lightning_logger
from lib.train.regression import RegressionTask, scalar_predictions
from lib.utils.log import get_logger
from lib.utils.utils import pick_device

type ModelInput = torch.Tensor | dict[str, torch.Tensor]

logger = get_logger()


class RegressionModule(L.LightningModule):
    def __init__(
        self, cfg: PytorchTrainConfig, model: nn.Module, task: RegressionTask | None = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        task = task or RegressionTask()
        self.criterion = task.criterion
        self.epoch_metrics = task.epoch_metrics
        self.history: list[dict[str, float | None]] = []
        self.epoch_lr = 0.0
        self._unfrozen = cfg.freeze_epochs == 0
        self._val_true: list[np.ndarray] = []
        self._val_pred: list[np.ndarray] = []
        self._train_true: list[np.ndarray] = []
        self._train_pred: list[np.ndarray] = []

    def forward(self, x: ModelInput) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        result: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = self.model(x)
        return result

    def _step(
        self, batch: tuple[ModelInput, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        if self.cfg.label_noise > 0 and self.training:
            y = y + torch.randn_like(y) * self.cfg.label_noise
        out = self.model(x)
        pred = out[0] if isinstance(out, tuple) else out
        y = scalar_predictions(y, len(y))
        pred = scalar_predictions(pred, len(y))
        if isinstance(out, tuple):
            loss = self.criterion(pred, y) + out[1]
        else:
            loss = self.criterion(pred, y)
        return loss, pred.detach(), y.detach()

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.cfg.freeze_epochs and not self._unfrozen:
            for p in self.model.parameters():
                p.requires_grad = True
            self._unfrozen = True
            self.trainer.strategy.setup_optimizers(self.trainer)
            logger.info(">>> Backbone unfrozen!")
        self.epoch_lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])

    def training_step(
        self, batch: tuple[ModelInput, torch.Tensor], batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, y = self._step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._train_pred.append(pred.cpu().numpy())
        self._train_true.append(y.cpu().numpy())
        return loss

    def validation_step(
        self, batch: tuple[ModelInput, torch.Tensor], batch_idx: int,
    ) -> None:
        loss, pred, y = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self._val_pred.append(pred.cpu().numpy())
        self._val_true.append(y.cpu().numpy())

    def on_validation_epoch_start(self) -> None:
        self._val_true = []
        self._val_pred = []

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train")
        record: dict[str, float | None] = {"epoch": self.current_epoch + 1, "lr": self.epoch_lr}
        for name, metric in self.trainer.callback_metrics.items():
            if name.startswith(("train/", "val/")) and metric.numel() == 1:
                value = float(metric.detach())
                record[name] = value if math.isfinite(value) else None
        self.history.append(record)
        self._train_true = []
        self._train_pred = []

    def _log_metrics(self, prefix: str) -> None:
        t_list = self._val_true if prefix == "val" else self._train_true
        p_list = self._val_pred if prefix == "val" else self._train_pred
        if not t_list:
            return
        y_true = pd.Series(np.concatenate(t_list).astype(np.float64))
        y_pred = np.concatenate(p_list).astype(np.float64)
        for key, value in self.epoch_metrics(y_true, y_pred).items():
            self.log(f"{prefix}/{key}", value)

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> tuple[list[AdamW], list[dict[str, object]]]:
        hp = self.cfg.head_prefix
        head = [p for n, p in self.model.named_parameters() if n.startswith(hp)]
        if self._unfrozen:
            rest = [p for n, p in self.model.named_parameters() if not n.startswith(hp)]
            params = [
                {"params": head, "lr": self.cfg.lr},
                {"params": rest, "lr": self.cfg.lr / 10},
            ]
            t_max = max(self.cfg.epochs - self.cfg.freeze_epochs, 1)
        else:
            params = [{"params": head, "lr": self.cfg.lr}]
            t_max = self.cfg.epochs
        opt = AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        sch = CosineAnnealingLR(opt, T_max=t_max)
        return [opt], [{"scheduler": sch, "interval": "epoch"}]


def _extract_model_state(ckpt_path: Path, out_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    torch.save(sd, out_path)


def run_regression(
    cfg: PytorchTrainConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    outdir: str,
    task: RegressionTask | None = None,
    device: torch.device | None = None,
) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(outdir)
    weights_dir = out / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device("auto")

    task = task or RegressionTask()
    module = RegressionModule(cfg, model, task)
    n_total = sum(p.numel() for p in module.model.parameters())
    n_train = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
    logger.info(f"[train] params: {n_total:,} total, {n_train:,} trainable")

    ckpt = ModelCheckpoint(
        dirpath=weights_dir,
        filename="best",
        monitor=task.monitor,
        mode=task.mode,
        save_top_k=1,
        save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=device.type,
        devices=[device.index or 0] if device.type == "cuda" else 1,
        callbacks=[ckpt],
        logger=lightning_logger(default=False),
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    (out / "history.json").write_text(json.dumps(module.history, indent=2, allow_nan=False))
    if ckpt.best_model_path:
        _extract_model_state(Path(ckpt.best_model_path), out / "best_model.pt")
    if ckpt.last_model_path:
        _extract_model_state(Path(ckpt.last_model_path), out / "last_model.pt")
    return out / "best_model.pt"
