import torch

from lib.models.targets import bounded_logit, scaled_sigmoid


def z_to_logistic(z: torch.Tensor, z_min: float = 0.0, z_max: float = 0.4) -> torch.Tensor:
    return bounded_logit(z, z_min, z_max)


def logistic_to_z(y: torch.Tensor, z_min: float = 0.0, z_max: float = 0.4) -> torch.Tensor:
    return scaled_sigmoid(y, z_min, z_max)
