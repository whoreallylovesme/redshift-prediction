import torch


def z_to_logistic(z: torch.Tensor, z_min: float = 0.0, z_max: float = 0.4) -> torch.Tensor:
    eps = 1e-6
    return torch.log((z - z_min).clamp(min=eps) / (z_max - z).clamp(min=eps))


def logistic_to_z(y: torch.Tensor, z_min: float = 0.0, z_max: float = 0.4) -> torch.Tensor:
    return z_min + (z_max - z_min) * torch.sigmoid(y)
