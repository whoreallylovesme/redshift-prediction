"""Directed weighted neighbor aggregation and residual messages."""

import torch
from torch import nn
from torch.nn import functional as F


def weighted_neighbors(
    features: torch.Tensor, index: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Compute one normalized weighted mean of source embeddings per receiving node."""
    if (
        features.ndim != 2
        or index.ndim != 2
        or index.dtype != torch.int64
        or weight.shape != index.shape
        or weight.dtype != features.dtype
        or not features.is_floating_point()
        or weight.device != features.device
        or index.device != features.device
        or (index < 0).any()
        or (index >= len(features)).any()
    ):
        raise ValueError("Expected a feature matrix and valid index and weight matrices")
    if not torch.isfinite(weight).all() or (weight < 0).any():
        raise ValueError("Neighbor weights must be finite and nonnegative")
    total = weight.sum(dim=1, keepdim=True)
    if not torch.isfinite(total).all() or (total <= 0).any():
        raise ValueError("Each receiver needs positive finite total neighbor weight")
    normalized = weight / total
    result = (features[index] * normalized[..., None]).sum(dim=1)
    if not torch.isfinite(result).all():
        raise ValueError("Unrepresentable weighted neighbor context")
    return result


class ResidualMessage(nn.Module):
    """h' = h + GELU(W_self h + W_context context + b), with shared embedding width."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.self_linear = nn.Linear(width, width)
        self.context_linear = nn.Linear(width, width, bias=False)

    def forward(self, own: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        first = self.self_linear.weight
        if (
            own.ndim != 2
            or own.shape[1] != self.self_linear.in_features
            or context.shape != own.shape
        ):
            raise ValueError("Each receiver and its context need the same embedding dimensions")
        if not torch.isfinite(own).all() or not torch.isfinite(context).all():
            raise ValueError("Message inputs must be finite")
        if any(x.dtype != first.dtype or x.device != first.device for x in (own, context)):
            raise ValueError("Message inputs must match parameter dtype/device")
        result = own + F.gelu(self.self_linear(own) + self.context_linear(context))
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite residual representation")
        return result
