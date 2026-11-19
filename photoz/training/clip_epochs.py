from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import lightning as L
import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, IterableDataset

from photoz.data.euclid_packet import schedule_lr


class ShuffledRowBatches(IterableDataset[torch.Tensor]):
    def __init__(self, rows: NDArray[np.int64], seed: int, size: int, drop_last: bool) -> None:
        self.rows = rows
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.drop_last = drop_last

    def __len__(self) -> int:
        return (len(self.rows) + (0 if self.drop_last else self.size - 1)) // self.size

    def __iter__(self) -> Iterator[torch.Tensor]:
        order = self.rng.permutation(self.rows)
        for start in range(0, len(self) * self.size, self.size):
            yield torch.from_numpy(order[start : start + self.size])

    def loader(self) -> DataLoader:
        if not len(self):
            raise ValueError("Training requires at least one batch")
        return DataLoader(self, batch_size=None, generator=torch.Generator())


class ClipTrainingEpoch(L.LightningModule):
    def __init__(self, epochs: int) -> None:
        super().__init__()
        self.epochs = epochs
        self.history: list[dict[str, Any]] = []
        self.loss_sum = 0.0
        self.examples = 0
        self.lr = 0.0

    def transfer_batch_to_device(
        self, batch: torch.Tensor, device: torch.device, dataloader_idx: int
    ) -> torch.Tensor:
        return batch

    def on_train_epoch_start(self) -> None:
        self.loss_sum = 0.0
        self.examples = 0
        self.lr = schedule_lr(self.current_epoch + 1, self.epochs)
        for group in self.trainer.optimizers[0].param_groups:
            group["lr"] = self.lr

    def record_loss(self, loss: torch.Tensor, size: int) -> None:
        self.loss_sum += float(loss.detach()) * size
        self.examples += size
        self.log("train/loss", loss, on_epoch=True, on_step=False, batch_size=size)
