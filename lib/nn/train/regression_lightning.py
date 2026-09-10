from __future__ import annotations

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

from lib.config_models import PytorchTrainConfig
from lib.log import get_logger
from lib.metrics import bias, nmad, outlier_percent
from lib.utils import pick_device

logger = get_logger()


class RegressionModule(L.LightningModule):
    def __init__(self, cfg: PytorchTrainConfig, model: nn.Module) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self._unfrozen = cfg.freeze_epochs == 0
        self._val_true: list[np.ndarray] = []
        self._val_pred: list[np.ndarray] = []
        self._train_true: list[np.ndarray] = []
        self._train_pred: list[np.ndarray] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        result: torch.Tensor | tuple[torch.Tensor, torch.Tensor] = self.model(x)
        return result

    @property
    def criterion(self) -> nn.Module:
        if self.cfg.z_weight_alpha > 0:
            from lib.nn.train.utils import WeightedHuberLoss

            return WeightedHuberLoss(delta=0.1, alpha=self.cfg.z_weight_alpha)
        return nn.HuberLoss(delta=0.1)

    def _step(
        self, batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        if self.cfg.label_noise > 0 and self.training:
            y = y + torch.randn_like(y) * self.cfg.label_noise
        out = self.model(x)
        pred = out[0] if isinstance(out, tuple) else out
        pred = pred.squeeze(-1)
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

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> torch.Tensor:
        loss, pred, y = self._step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._train_pred.append(pred.cpu().numpy())
        self._train_true.append(y.cpu().numpy())
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> None:
        loss, pred, y = self._step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self._val_pred.append(pred.cpu().numpy())
        self._val_true.append(y.cpu().numpy())

    def on_validation_epoch_start(self) -> None:
        self._val_true = []
        self._val_pred = []

    def on_validation_epoch_end(self) -> None:
        self._log_z_metrics("val")

    def on_train_epoch_end(self) -> None:
        self._log_z_metrics("train")
        self._train_true = []
        self._train_pred = []

    def _log_z_metrics(self, prefix: str) -> None:
        t_list = self._val_true if prefix == "val" else self._train_true
        p_list = self._val_pred if prefix == "val" else self._train_pred
        if not t_list:
            return
        y_true = pd.Series(np.concatenate(t_list).astype(np.float64))
        y_pred = np.concatenate(p_list).astype(np.float64)
        self.log(f"{prefix}/nmad", float(nmad(y_true, y_pred)))
        self.log(f"{prefix}/bias", float(bias(y_true, y_pred)))
        self.log(f"{prefix}/outlier_pct", float(outlier_percent(y_true, y_pred)))

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
) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(outdir)
    weights_dir = out / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    accelerator = "mps" if pick_device("auto").type == "mps" else "auto"

    module = RegressionModule(cfg, model)
    n_total = sum(p.numel() for p in module.model.parameters())
    n_train = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
    logger.info(f"[train] params: {n_total:,} total, {n_train:,} trainable")

    ckpt = ModelCheckpoint(
        dirpath=weights_dir,
        filename="best",
        monitor="val/nmad",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=accelerator,
        devices=1,
        callbacks=[ckpt],
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best = weights_dir / "best.ckpt"
    if best.exists():
        _extract_model_state(best, out / "best_model.pt")
    last = weights_dir / "last.ckpt"
    if last.exists():
        _extract_model_state(last, out / "last_model.pt")
    return out / "best_model.pt"
