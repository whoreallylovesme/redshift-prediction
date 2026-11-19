"""Masked observation sampling and heteroscedastic reconstruction likelihood."""

import math
from dataclasses import dataclass

import torch


def _valid_tensor_mask(valid: torch.Tensor) -> None:
    if valid.ndim != 2 or valid.dtype != torch.bool or valid.shape[1] == 0:
        raise ValueError("valid must be a nonzero-width [rows,features] boolean tensor")


def check_mask(valid: torch.Tensor, masked: torch.Tensor) -> None:
    _valid_tensor_mask(valid)
    if masked.dtype != torch.bool or masked.shape != valid.shape or masked.device != valid.device:
        raise ValueError("masked must be a boolean tensor matching validity shape/device")
    if (masked & ~valid).any():
        raise ValueError("Cannot mask an invalid/unobserved measurement")
    active = masked.any(dim=1)
    if (active & ((valid.sum(dim=1) < 2) | (~masked & valid).sum(dim=1).eq(0))).any():
        raise ValueError("Every masked row needs at least one hidden and one visible measurement")


def sample_mask(
    valid: torch.Tensor,
    *,
    fraction: float = 0.25,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Hide round-half-up(fraction * measured), clamped to [1,n-1] for n>=2.

    Rows with zero/one measurements stay in output with no SSL target. A
    four-feature row with fraction=.25 hides exactly one measurement.
    """
    _valid_tensor_mask(valid)
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Mask fraction must be finite and strictly between zero and one")
    counts = valid.sum(dim=1)
    hidden_count = torch.floor(counts.to(torch.float64) * fraction + 0.5).to(torch.int64)
    hidden_count = torch.minimum(hidden_count.clamp_min(1), (counts - 1).clamp_min(0))
    noise = torch.rand(valid.shape, device=valid.device, generator=generator)
    noise = noise.masked_fill(~valid, float("inf"))
    ranks = torch.argsort(torch.argsort(noise, dim=1), dim=1)
    return valid & (ranks < hidden_count.unsqueeze(1))


@dataclass(frozen=True)
class MaskedPrediction:
    """Conditional mean, intrinsic variance and the representation of the observed inputs."""

    mean: torch.Tensor
    intrinsic_variance: torch.Tensor
    embedding: torch.Tensor


def masked_gaussian_nll(
    prediction: MaskedPrediction,
    target_value: torch.Tensor,
    target_error: torch.Tensor,
    valid: torch.Tensor,
    masked: torch.Tensor,
) -> torch.Tensor:
    """Mean within each masked row, then mean over active rows only.

    Original targets/errors must be supplied, not zeroed encoder inputs. Targets
    are detached and selected *before* arithmetic, so unused NaNs cannot poison
    the loss or its gradients. Computation uses float64 to avoid float32 squaring
    overflow. Empty objectives are differentiable zero, not NaN or fake samples.
    """
    check_mask(valid, masked)
    tensors = (prediction.mean, prediction.intrinsic_variance, target_value, target_error)
    if any(
        tensor.shape != valid.shape
        or tensor.device != valid.device
        or not tensor.is_floating_point()
        for tensor in tensors
    ):
        raise ValueError("NLL inputs need matching [rows,features] floating tensors/device")
    selected = valid & masked
    if not selected.any():
        return prediction.mean.reshape(-1)[:0].sum() + (
            prediction.intrinsic_variance.reshape(-1)[:0].sum()
        )
    mean = prediction.mean[selected].to(torch.float64)
    intrinsic = prediction.intrinsic_variance[selected].to(torch.float64)
    target = target_value.detach()[selected].to(torch.float64)
    error = target_error.detach()[selected].to(torch.float64)
    if (
        not all(torch.isfinite(value).all() for value in (mean, intrinsic, target, error))
        or (intrinsic <= 0).any()
        or (error <= 0).any()
    ):
        raise ValueError(
            "Selected targets/predictions must be finite with positive variances/errors"
        )
    variance = intrinsic + error.square()
    terms = 0.5 * ((target - mean).square() / variance + variance.log() + math.log(2 * math.pi))
    if not torch.isfinite(terms).all():
        raise ValueError("Nonfinite Gaussian NLL; inspect value scales and measurement errors")
    row_indices = selected.nonzero(as_tuple=True)[0]
    totals = torch.zeros(len(valid), dtype=terms.dtype, device=terms.device)
    totals = totals.index_add(0, row_indices, terms)
    counts = selected.sum(dim=1)
    return (totals[counts > 0] / counts[counts > 0]).mean()
