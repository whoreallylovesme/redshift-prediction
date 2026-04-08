import torch
import torch.nn as nn


class WeightedHuberLoss(nn.Module):

    def __init__(self, delta: float = 0.1, alpha: float = 2.0) -> None:
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="none")
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.huber(pred, target)
        weights = 1.0 + self.alpha * target
        result: torch.Tensor = (loss * weights).mean()
        return result


def unfreeze_backbone(model: nn.Module) -> None:

    for param in model.parameters():
        param.requires_grad = True
