"""Instrument-token encoder and Gaussian likelihood for masked native photometry."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from lib.models.encoders import attention_stack
from lib.models.masked import MaskedPrediction as MaskedFluxPrediction
from lib.models.masked import check_mask
from lib.models.masked import masked_gaussian_nll as _gaussian_nll
from lib.models.masked import sample_mask as sample_mask

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
        self.transformer = attention_stack(
            num_layers, width=hidden_dim, heads=num_heads, dropout=dropout, norm_first=False
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
        check_mask(valid, masked)
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
    return _gaussian_nll(prediction, target_flux, target_error, valid, masked)
