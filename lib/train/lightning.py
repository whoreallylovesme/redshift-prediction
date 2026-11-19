from __future__ import annotations

from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from lib.tracking import lightning_logger


def single_device_trainer(
    device: torch.device, epochs: int, *, gradient_clip: float | None = None, track: bool = True
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
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )


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
