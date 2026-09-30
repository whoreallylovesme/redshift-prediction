from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from lib.metrics import bias, nmad, outlier_fraction
from lib.models.capsnet import CapsNet, logistic_to_z, margin_loss, mdn_pz, z_to_logistic
from lib.utils.log import get_logger
from lib.utils.utils import pick_device

logger = get_logger()


@dataclass
class TrainConfig:
    out_dir: str
    in_channels: int = 5
    crop_size: int = 64
    stage_a_epochs: int = 30
    stage_b_epochs: int = 100
    lr: float = 5e-4
    decay_rate: float = 0.95
    lam_recon: float = 0.005
    lam_redshift: float = 1.0
    z_min: float = 0.0
    z_max: float = 0.4
    outlier_threshold: float = 0.15
    mdn_components: int = 0
    redshift_loss: str = "mse"
    lr_schedule: str = "exp"
    seed: int = 200
    device: str = "auto"


class StopTrainingRequested(Exception):
    pass


_TRAIN_CFG_FIELDS = {f for f in TrainConfig.__dataclass_fields__}



class StopFile(L.Callback):
    def __init__(self, stop_path: Path) -> None:
        self.stop_path = stop_path

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
        batch: object, batch_idx: int,
    ) -> None:
        if self.stop_path.exists():
            raise KeyboardInterrupt


class JsonlLogger(L.Callback):
    def __init__(self, log_path: Path, stage: str) -> None:
        self.log_path = log_path
        self.stage = stage
        self.t0 = 0.0

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
    ) -> None:
        self.t0 = time.time()

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
    ) -> None:
        m = {k: float(v) for k, v in trainer.callback_metrics.items()}
        rec = {
            "stage": self.stage,
            "epoch": int(trainer.current_epoch) + 1,
            "train": {k.split("/")[-1]: v for k, v in m.items() if k.startswith("train/")},
            "val": {k.split("/")[-1]: v for k, v in m.items() if k.startswith("val/")},
            "lr": m.get("lr", float("nan")),
            "dt": time.time() - self.t0,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")


class CapsNetModule(L.LightningModule):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = CapsNet(
            in_channels=cfg.in_channels, input_size=cfg.crop_size,
            num_class=2, dim_capsule=16, routings=3,
            mdn_components=cfg.mdn_components,
        )
        self.stage = "B"
        self.num_pixels = cfg.in_channels * cfg.crop_size * cfg.crop_size
        self._optimizer_state: dict[str, Any] | None = None
        self._scheduler_state: dict[str, Any] | None = None
        self._val_z_true: list[np.ndarray] = []
        self._val_z_pred: list[np.ndarray] = []

    def forward(
        self, images: torch.Tensor, target_class: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = self.net(
            images, target_class=target_class,
        )
        return result

    def _losses(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], np.ndarray, np.ndarray] | None:
        images, z, morph = batch
        device = self.device
        images = images.to(device, non_blocking=True)
        z = z.to(device, non_blocking=True)
        morph = morph.to(device, non_blocking=True)
        cfg = self.cfg

        labeled = ~torch.isnan(morph)
        if self.stage == "A" and not labeled.any():
            return None

        true_cls = morph.long()
        if self.training:
            target = torch.where(labeled, true_cls, torch.full_like(true_cls, -1))
        else:
            target = torch.full_like(true_cls, -1)

        z_logistic_pred, probs, recon = self.net(images, target_class=target)

        if labeled.any():
            onehot = F.one_hot(true_cls[labeled], num_classes=self.net.num_class).float()
            l_margin = margin_loss(probs[labeled], onehot)
        else:
            l_margin = torch.zeros((), device=device)
        l_recon = F.mse_loss(recon, images)

        if self.stage == "B":
            z_clamped = z.clamp(min=cfg.z_min + 1e-5, max=cfg.z_max - 1e-5)
            y_true = z_to_logistic(z_clamped, cfg.z_min, cfg.z_max)
            if cfg.mdn_components > 0:
                from lib.models.capsnet import mdn_nll

                l_redshift = mdn_nll(z_logistic_pred, y_true)
            elif cfg.redshift_loss == "huber":
                l_redshift = F.smooth_l1_loss(z_logistic_pred, y_true)
            else:
                l_redshift = F.mse_loss(z_logistic_pred, y_true)
        else:
            l_redshift = torch.zeros((), device=device)

        total = (
            l_margin
            + cfg.lam_recon * self.num_pixels * l_recon
            + cfg.lam_redshift * l_redshift
        )
        parts = {"margin": l_margin, "recon": l_recon, "redshift": l_redshift}
        raw = z_logistic_pred.detach()
        if cfg.mdn_components > 0:
            grid_y = torch.linspace(0.01, 0.99, 151, device=raw.device)
            pz = mdn_pz(raw, grid_y)  # [B,G] в логистическом пространстве
            z_vals = logistic_to_z(grid_y, cfg.z_min, cfg.z_max)
            z_pred_phys = z_vals[pz.argmax(dim=1)]
        else:
            z_pred_phys = logistic_to_z(raw, cfg.z_min, cfg.z_max)
        return total, parts, z.detach().cpu().numpy(), z_pred_phys.cpu().numpy()

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> torch.Tensor | None:
        res = self._losses(batch)
        if res is None:
            return None
        total, parts, _, _ = res
        logs = {"train/" + k: float(v.item()) for k, v in parts.items()}
        logs["train/total"] = float(total.item())
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False)
        opts = self.optimizers()
        opt = opts[0] if isinstance(opts, list) else opts
        self.log(
            "lr",
            opt.param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
        )
        return total

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int,
    ) -> None:
        res = self._losses(batch)
        if res is None:
            return
        total, parts, z_true, z_pred = res
        logs = {"val/" + k: float(v.item()) for k, v in parts.items()}
        logs["val/total"] = float(total.item())
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=False)
        self._val_z_true.append(z_true)
        self._val_z_pred.append(z_pred)

    def on_validation_epoch_start(self) -> None:
        self._val_z_true = []
        self._val_z_pred = []

    def on_validation_epoch_end(self) -> None:
        if not self._val_z_true:
            return
        z_true = np.concatenate(self._val_z_true)
        z_pred = np.concatenate(self._val_z_pred)
        s_true = pd.Series(z_true.astype(np.float64))
        p_pred = z_pred.astype(np.float64)
        self.log("val/nmad", float(nmad(s_true, p_pred)))
        self.log("val/bias", float(bias(s_true, p_pred)))
        self.log(
            "val/outlier_frac",
            float(outlier_fraction(s_true, p_pred, self.cfg.outlier_threshold)),
        )

    def configure_optimizers(  # type: ignore[override]
        self,
    ) -> tuple[list[Adam], list[dict[str, object]]]:
        opt = Adam(self.parameters(), lr=self.cfg.lr)
        if self._optimizer_state is not None:
            opt.load_state_dict(self._optimizer_state)
        sch: ExponentialLR | torch.optim.lr_scheduler.CosineAnnealingLR
        if self.cfg.lr_schedule == "cosine":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(int(self.cfg.stage_b_epochs), 1),
            )
        else:
            sch = ExponentialLR(opt, gamma=self.cfg.decay_rate)
        if self._scheduler_state is not None:
            sch.load_state_dict(self._scheduler_state)
        return [opt], [{"scheduler": sch, "interval": "epoch"}]


def _extract_net_state(ckpt_path: Path, out_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = {
        k.removeprefix("net."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("net.")
    }
    torch.save(sd, out_path)


def run_staged(
    module: CapsNetModule,
    train_a: DataLoader | None,
    val_a: DataLoader | None,
    train_b: DataLoader,
    val_b: DataLoader,
    cfg: TrainConfig,
) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    stop_path = weights_dir / "STOP"
    log_path = out_dir / "log.jsonl"
    accelerator = "mps" if pick_device(cfg.device).type == "mps" else "auto"

    def make_trainer(stage: str, monitor: str | None) -> L.Trainer:
        callbacks: list[L.Callback] = [
            StopFile(stop_path),
            JsonlLogger(log_path, stage),
        ]
        if monitor is not None:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=weights_dir,
                    filename=f"stage_{stage.lower()}_best",
                    monitor=monitor,
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                ),
            )
        return L.Trainer(
            max_epochs=(cfg.stage_a_epochs if stage == "A" else cfg.stage_b_epochs),
            accelerator=accelerator,
            devices=1,
            callbacks=callbacks,
            logger=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            deterministic=False,
        )

    if cfg.stage_a_epochs > 0 and train_a is not None and val_a is not None:
        module.stage = "A"
        trainer_a = make_trainer("A", "val/margin")
        trainer_a.fit(module, train_dataloaders=train_a, val_dataloaders=val_a)
        module._optimizer_state = trainer_a.optimizers[0].state_dict()
        schedulers = trainer_a.lr_scheduler_configs
        module._scheduler_state = schedulers[0].scheduler.state_dict()
        best_a = weights_dir / "stage_a_best.ckpt"
        if best_a.exists():
            _extract_net_state(best_a, weights_dir / "stage_a_best.pt")
        last_a = weights_dir / "last.ckpt"
        if last_a.exists():
            _extract_net_state(last_a, weights_dir / "stage_a_last.pt")

    module.stage = "B"
    trainer_b = make_trainer("B", "val/nmad")
    trainer_b.fit(module, train_dataloaders=train_b, val_dataloaders=val_b)
    module._optimizer_state = trainer_b.optimizers[0].state_dict()
    schedulers_b = trainer_b.lr_scheduler_configs
    module._scheduler_state = schedulers_b[0].scheduler.state_dict()
    best_b = weights_dir / "stage_b_best.ckpt"
    if best_b.exists():
        _extract_net_state(best_b, weights_dir / "stage_b_best.pt")
    last_b = weights_dir / "last.ckpt"
    if last_b.exists():
        _extract_net_state(last_b, weights_dir / "stage_b_last.pt")

    return weights_dir / "stage_b_last.pt"
