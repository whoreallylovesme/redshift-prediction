from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from lib.dataloaders.batches import ShuffledRowBatches as ShuffledRowBatches
from photoz.data.euclid_packet import WARMUP_EPOCHS, schedule_lr

type OptimizerSchedule = tuple[list[AdamW], list[LambdaLR]]


def scheduled_adamw(
    parameters: Iterable[torch.Tensor], epochs: int, *, lr: float,
    weight_decay: float, warmup: int = WARMUP_EPOCHS,
) -> OptimizerSchedule:
    optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = LambdaLR(
        optimizer, lambda step: schedule_lr(min(step + 1, epochs), epochs, 1.0, warmup)
    )
    return [optimizer], [scheduler]


class ClipTrainingEpoch(L.LightningModule):
    def __init__(self, epochs: int) -> None:
        super().__init__()
        self.epochs = epochs
        self.history: list[dict[str, Any]] = []
        self.examples = 0
        self.lr = 0.0

    def transfer_batch_to_device(
        self, batch: torch.Tensor, device: torch.device, dataloader_idx: int
    ) -> torch.Tensor:
        return batch

    def on_train_epoch_start(self) -> None:
        self.examples = 0
        self.lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])

    def record_loss(self, loss: torch.Tensor, size: int) -> None:
        self.examples += size
        value = loss.detach()
        if value.device.type != "mps":
            value = value.double()
        self.log("train/loss", value, on_epoch=True, on_step=False, batch_size=size)
