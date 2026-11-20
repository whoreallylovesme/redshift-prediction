import torch


def bounded_logit(target: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    if not lower < upper:
        raise ValueError("Target bounds must be increasing")
    return torch.log((target - lower).clamp_min(1e-6) / (upper - target).clamp_min(1e-6))


def scaled_sigmoid(logits: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    if not lower < upper:
        raise ValueError("Target bounds must be increasing")
    return lower + (upper - lower) * torch.sigmoid(logits)
