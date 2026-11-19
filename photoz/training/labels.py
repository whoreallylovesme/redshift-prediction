from __future__ import annotations

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from lib.models.nn.capsnet import CapsNet, margin_loss
from lib.train.lightning import fit_best, predict_batches

_T = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class MorphologyClassifier(L.LightningModule):
    def __init__(
        self, model: CapsNet, lr: float, decay_rate: float, lam_recon: float, num_pixels: int
    ) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.decay_rate = decay_rate
        self.reconstruction_weight = lam_recon * num_pixels
        self.accuracy = MulticlassAccuracy(num_classes=model.num_class, average="micro")

    def configure_optimizers(self) -> tuple[list[Adam], list[ExponentialLR]]:
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return [optimizer], [ExponentialLR(optimizer, gamma=self.decay_rate)]

    def training_step(self, batch: _T, batch_idx: int) -> torch.Tensor:
        images, _, morph = batch
        _, probs, recon = self.model(images, target_class=morph.long())
        onehot = F.one_hot(morph.long(), num_classes=self.model.num_class).float()
        loss = margin_loss(probs, onehot) + self.reconstruction_weight * F.mse_loss(recon, images)
        self.log("train/loss", loss, on_epoch=True, on_step=False, batch_size=len(images))
        return loss

    def validation_step(self, batch: _T, batch_idx: int) -> None:
        images, _, morph = batch
        _, probs, _ = self.model(images, target_class=None)
        self.accuracy.update(probs, morph.long())
        self.log("val/accuracy", self.accuracy)


def _train_classifier(
    model: CapsNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    decay_rate: float,
    lam_recon: float,
    num_pixels: int,
) -> None:
    module = MorphologyClassifier(model, lr, decay_rate, lam_recon, num_pixels)
    if epochs and len(train_loader):
        fit_best(
            module, train_loader, val_loader, device, epochs, monitor="val/accuracy", mode="max"
        )
    model.to(device)


class MorphologyProbabilities(L.LightningModule):
    def __init__(self, model: CapsNet) -> None:
        super().__init__()
        self.model = model

    def predict_step(self, batch: _T, batch_idx: int) -> torch.Tensor:
        images, _, _ = batch
        _, probabilities, _ = self.model(images, target_class=None)
        predicted: torch.Tensor = probabilities[:, 1]
        return predicted


def _predict_probs(model: CapsNet, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    result = predict_batches(MorphologyProbabilities(model), loader, device)
    return torch.cat(result).cpu().numpy() if result else np.empty(0, dtype=np.float32)
