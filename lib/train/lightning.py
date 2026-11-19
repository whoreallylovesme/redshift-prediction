from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from lib.tracking.mlflow import lightning_logger


def single_device_trainer(
    device: torch.device, epochs: int, *, gradient_clip: float | None = None, track: bool = True,
    callbacks: list[L.Callback] | None = None,
) -> L.Trainer:
    devices = [device.index or 0] if device.type == "cuda" else 1
    return L.Trainer(
        accelerator=device.type,
        devices=devices,
        max_epochs=epochs,
        gradient_clip_val=gradient_clip,
        benchmark=torch.backends.cudnn.benchmark,
        num_sanity_val_steps=0,
        logger=lightning_logger(default=False) if track else False,
        callbacks=callbacks,
        enable_checkpointing=any(isinstance(c, ModelCheckpoint) for c in callbacks or []),
        enable_model_summary=False,
        enable_progress_bar=False,
    )


def fit_best(
    module: L.LightningModule, train: DataLoader, valid: DataLoader, device: torch.device,
    epochs: int, *, monitor: str, mode: Literal["min", "max"] = "min",
    gradient_clip: float | None = None, export_dir: Path | None = None,
) -> tuple[int, float]:
    """Fit and restore the selected epoch, optionally exporting native model weights."""
    with TemporaryDirectory(dir=export_dir) as directory:
        best = ModelCheckpoint(
            dirpath=directory, filename="best", monitor=monitor, mode=mode,
            save_weights_only=True, enable_version_counter=False,
        )
        trainer = single_device_trainer(
            device, epochs, gradient_clip=gradient_clip, callbacks=[best]
        )
        trainer.fit(module, train, valid)
        score = best.best_model_score
        if not best.best_model_path or score is None or not torch.isfinite(score):
            raise ValueError(f"No finite validation checkpoint for {monitor}")
        if export_dir is not None:
            torch.save(module.get_submodule("model").state_dict(), export_dir / "last_state.pt")
        checkpoint = torch.load(best.best_model_path, map_location="cpu", weights_only=True)
        module.load_state_dict(checkpoint["state_dict"])
        if export_dir is not None:
            torch.save(module.get_submodule("model").state_dict(), export_dir / "selected_state.pt")
        return int(checkpoint["epoch"]) + 1, float(score)


def export_weights(checkpoint_path: Path, output: Path, prefix: str) -> None:
    """Export the wrapped model's native state_dict from a Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {k.removeprefix(prefix): v for k, v in checkpoint["state_dict"].items()
             if k.startswith(prefix)}
    torch.save(state, output)


def predict_batches(
    module: L.LightningModule, loader: DataLoader, device: torch.device
) -> list[Any]:
    if not len(loader):
        return []
    trainer = single_device_trainer(device, 1, track=False)
    try:
        results = trainer.predict(module, dataloaders=loader, return_predictions=True)
        if results is None:
            raise RuntimeError("Lightning did not return predictions")
        return list(results)
    finally:
        module.to(device)
