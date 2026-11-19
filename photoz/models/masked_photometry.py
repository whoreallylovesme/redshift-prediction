"""Instrument-token encoder and Gaussian likelihood for masked native photometry."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

COMMON_BANDS = ("VIS", "Y", "J", "H")
_VERSION = 1


def _bands(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError("Bands must be an ordered sequence, not one string")
    bands = tuple(values)
    prohibited = {"ra", "dec", "object_id", "phz", "specz", "redshift", "z_spec"}
    if (
        not bands
        or any(not isinstance(value, str) or not value.strip() for value in bands)
        or len(set(bands)) != len(bands)
        or any(value.lower() in prohibited for value in bands)
    ):
        raise ValueError("Bands must be distinct instrument/passband names, not truth or geometry")
    return bands


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _schema_sha256(bands: tuple[str, ...], flux_unit: str) -> str:
    return _json_sha256(
        {
            "version": _VERSION,
            "bands": bands,
            "flux_unit": flux_unit,
            "valid": "finite flux and finite strictly positive error",
            "transform": "raw flux/scale; raw error/scale; explicit validity mask",
        }
    )


def _valid_tensor_mask(valid: torch.Tensor) -> None:
    if valid.ndim != 2 or valid.dtype != torch.bool or valid.shape[1] == 0:
        raise ValueError("valid must be a nonzero-band [rows,bands] boolean tensor")


def _check_mask(valid: torch.Tensor, masked: torch.Tensor) -> None:
    _valid_tensor_mask(valid)
    if masked.dtype != torch.bool or masked.shape != valid.shape or masked.device != valid.device:
        raise ValueError("masked must be a boolean tensor matching validity shape/device")
    if (masked & ~valid).any():
        raise ValueError("Cannot mask an invalid/unmeasured band")
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
    four-measurement row with fraction=.25 hides exactly one measurement.
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
class MaskedFluxPrediction:
    """Predictions in saved scaled-raw-flux units; CLS embedding is target-free."""

    mean: torch.Tensor
    intrinsic_variance: torch.Tensor
    embedding: torch.Tensor


class MaskedFluxEncoder(nn.Module):
    """Small instrument-token Transformer with CLS and explicit visibility flags.

    Both hidden flux AND hidden error are removed before any feature transform.
    Known target errors enter only the reconstruction loss. Missing rows are
    retained and finite, not promoted to successful measured reconstructions.
    """

    def __init__(
        self,
        bands: Sequence[str] = COMMON_BANDS,
        *,
        hidden_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
        variance_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        self.bands = _bands(bands)
        if (
            any(
                type(value) is not int or value <= 0
                for value in (hidden_dim, num_heads, num_layers)
            )
            or hidden_dim % num_heads != 0
            or not math.isfinite(dropout)
            or not 0 <= dropout < 1
            or not math.isfinite(variance_floor)
            or variance_floor <= 0
        ):
            raise ValueError("Invalid encoder dimensions/dropout/variance floor")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.variance_floor = variance_floor
        self.measurement_projection = nn.Linear(5, hidden_dim)
        self.instrument_embedding = nn.Embedding(len(self.bands), hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=2 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.decoder = nn.Linear(hidden_dim, 2)

    def config_dict(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "bands": list(self.bands),
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "variance_floor": self.variance_floor,
            "schema_sha256": _schema_sha256(self.bands, "saved-normalizer-scaled-raw"),
        }

    def get_extra_state(self) -> dict[str, Any]:
        return self.config_dict()

    def set_extra_state(self, state: Any) -> None:
        if state != self.config_dict():
            raise ValueError("Encoder checkpoint schema/config mismatch")

    def forward(
        self,
        flux: torch.Tensor,
        error: torch.Tensor,
        valid: torch.Tensor,
        *,
        masked: torch.Tensor,
        bands: Sequence[str],
    ) -> MaskedFluxPrediction:
        if _bands(bands) != self.bands:
            raise ValueError("Band schema/order mismatch in encoder")
        _check_mask(valid, masked)
        if (
            flux.shape != valid.shape
            or error.shape != valid.shape
            or valid.shape[1] != len(self.bands)
            or not flux.is_floating_point()
            or error.dtype != flux.dtype
            or flux.device != valid.device
            or error.device != valid.device
            or flux.dtype != self.cls.dtype
            or flux.device != self.cls.device
        ):
            raise ValueError("Flux/error shape, floating dtype or device mismatch")
        visible = valid & ~masked
        safe_f = torch.where(visible, flux, 0.0)
        safe_e = torch.where(visible, error, 0.0)
        if (
            not torch.isfinite(safe_f).all()
            or not torch.isfinite(safe_e).all()
            or (safe_e[visible] <= 0).any()
        ):
            raise ValueError("Visible measurements need finite flux and finite positive error")
        features = torch.stack(
            [
                torch.asinh(safe_f),
                torch.log1p(safe_e),
                visible.to(flux.dtype),
                masked.to(flux.dtype),
                (~valid).to(flux.dtype),
            ],
            dim=-1,
        )
        indices = torch.arange(len(self.bands), device=flux.device)
        tokens = self.measurement_projection(features) + self.instrument_embedding(indices)[None]
        cls = self.cls.expand(len(flux), -1, -1)
        sequence = self.transformer(torch.cat([cls, tokens], dim=1))
        decoded: torch.Tensor = self.decoder(sequence[:, 1:])
        return MaskedFluxPrediction(
            mean=decoded[..., 0],
            intrinsic_variance=F.softplus(decoded[..., 1]) + self.variance_floor,
            embedding=sequence[:, 0],
        )


def masked_gaussian_nll(
    prediction: MaskedFluxPrediction,
    target_flux: torch.Tensor,
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
    _check_mask(valid, masked)
    tensors = (prediction.mean, prediction.intrinsic_variance, target_flux, target_error)
    if any(
        tensor.shape != valid.shape
        or tensor.device != valid.device
        or not tensor.is_floating_point()
        for tensor in tensors
    ):
        raise ValueError("NLL inputs need matching [rows,bands] floating tensors/device")
    selected = valid & masked
    if not selected.any():
        return prediction.mean.reshape(-1)[:0].sum() + (
            prediction.intrinsic_variance.reshape(-1)[:0].sum()
        )
    mean = prediction.mean[selected].to(torch.float64)
    intrinsic = prediction.intrinsic_variance[selected].to(torch.float64)
    target = target_flux.detach()[selected].to(torch.float64)
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
        raise ValueError("Nonfinite Gaussian NLL; inspect native scales/measurement errors")
    row_indices = selected.nonzero(as_tuple=True)[0]
    totals = torch.zeros(len(valid), dtype=terms.dtype, device=terms.device)
    totals = totals.index_add(0, row_indices, terms)
    counts = selected.sum(dim=1)
    return (totals[counts > 0] / counts[counts > 0]).mean()
