from pathlib import Path

import lightning as L
import numpy as np
import torch
from torch import Tensor, nn

from photoz.models.nn.euclid_clip import CLIPModel, ITNetwork

SSL_EPOCHS = 12
FINE_EPOCHS = 40
SSL_BATCH = 256
FINE_BATCH = 128
INFER_BATCH = 512
LR = 3e-4
WD = 1e-4
CLIP_NORM = 5.0
WARMUP_EPOCHS = 5


def reset_rng(seed: int) -> None:
    L.seed_everything(seed, workers=False, verbose=False)


def state_copy(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def schedule_lr(
    epoch: int, epochs: int, base_lr: float = LR, warmup: int = WARMUP_EPOCHS
) -> float:
    if not 1 <= epoch <= epochs or warmup < 0:
        raise ValueError("schedule epoch outside bounds")
    if epoch <= warmup:
        return base_lr * epoch / warmup
    return float(base_lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (epochs - warmup))))


def masked(b: dict[str, Tensor], keep: Tensor) -> dict[str, Tensor]:
    m = b["table_mask"] & keep
    b["table_mask"] = m
    b["table_available"] = b["table_available"] & (m.sum(1) >= 2)
    return b


def load_it_from_clip(path: Path) -> ITNetwork:
    state = torch.load(path, map_location="cpu", weights_only=True)
    clip = CLIPModel(ITNetwork())
    clip.load_state_dict(state, strict=True)
    return clip.it_network
