"""Piecewise-uniform redshift density, proper scoring and quantiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn
from torch.nn import functional as F

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_VERSION = 1


def _real_array(value: ArrayLike, name: str, ndim: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.ndim != ndim:
        raise ValueError(f"{name} must be a real numeric {ndim}-D array")
    return np.array(raw, dtype=np.float64, copy=True)


def _edges(value: ArrayLike) -> FloatArray:
    edges = _real_array(value, "z_edges", 1)
    with np.errstate(over="ignore", invalid="ignore"):
        widths = np.diff(edges)
    if (
        len(edges) < 2
        or not np.isfinite(edges).all()
        or edges[0] <= -1
        or not np.isfinite(widths).all()
        or not (widths > 0).all()
    ):
        raise ValueError("z_edges must be finite, strictly increasing and start above -1")
    return edges


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class DensityPrediction:
    """Torch outputs in physical z units; mass sums to one, density need not."""

    logits: torch.Tensor
    log_bin_mass: torch.Tensor
    bin_mass: torch.Tensor
    density: torch.Tensor


class PhotoZDensityHead(nn.Module):
    """Same head for cold-supervised and SSL encoders, with config custody.

    Its inputs are already constructed target-free embeddings. No normalization,
    fitting or upstream support/identity certification is performed here. All
    probabilities/densities are computed in float64 to retain extreme log-mass
    values even when their exponential underflows to numerical zero.
    """

    def __init__(
        self,
        embedding_dim: int,
        z_edges: ArrayLike,
        *,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = _positive_int(embedding_dim, "embedding_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        if isinstance(dropout, bool) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be finite in [0, 1)")
        self.dropout = float(dropout)
        self._z_edges = tuple(float(item) for item in _edges(z_edges))
        self.layers = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, len(self._z_edges) - 1),
        )

    @property
    def z_edges(self) -> FloatArray:
        """Defensive copy of the immutable physical-redshift axis."""
        return np.array(self._z_edges, dtype=np.float64)

    def config_dict(self) -> dict[str, Any]:
        return {
            "version": _VERSION,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "z_edges": list(self._z_edges),
            "output": "categorical bin mass; piecewise-uniform density in physical z",
            "point_estimate": "interpolated piecewise-uniform CDF median",
        }

    def get_extra_state(self) -> dict[str, Any]:
        return self.config_dict()

    def set_extra_state(self, state: Any) -> None:
        if state != self.config_dict():
            raise ValueError("Photo-z head checkpoint config/redshift-axis mismatch")

    def forward(self, embedding: torch.Tensor) -> DensityPrediction:
        first = next(self.layers.parameters())
        if (
            embedding.ndim != 2
            or embedding.shape[1] != self.embedding_dim
            or not embedding.is_floating_point()
            or embedding.dtype != first.dtype
            or embedding.device != first.device
            or not torch.isfinite(embedding).all()
        ):
            raise ValueError("Embedding needs finite [rows,features] and head dtype/device")
        logits: torch.Tensor = self.layers(embedding)
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite photo-z logits")
        log_mass = F.log_softmax(logits.to(torch.float64), dim=1)
        mass = log_mass.exp()
        widths = torch.tensor(np.diff(self.z_edges), dtype=torch.float64, device=logits.device)
        density = mass / widths
        if not torch.isfinite(log_mass).all() or not torch.isfinite(density).all():
            raise ValueError("Unrepresentable photo-z log mass/density; inspect grid and logits")
        return DensityPrediction(logits, log_mass, mass, density)


def piecewise_uniform_nll(
    logits: torch.Tensor,
    reference_z: torch.Tensor,
    z_edges: ArrayLike,
    *,
    selected: torch.Tensor | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """Proper density NLL: ``-log_softmax(logits)[bin] + log(bin_width)``.

    Bins are [left,right), except the final bin includes its right endpoint.
    Selected nonfinite or out-of-grid truth raises, never clips/drops a target.
    Unselected logits/targets are indexed out BEFORE arithmetic, so NaNs cannot
    poison gradients. An empty objective requires an explicit all-false mask;
    its differentiable zero is a technical empty loss, not a successful sample.
    ``reduction='none'`` returns selected-row terms in their original order.
    """
    edges = _edges(z_edges)
    if (
        logits.ndim != 2
        or logits.shape[1] != len(edges) - 1
        or not logits.is_floating_point()
        or reference_z.shape != (len(logits),)
        or not reference_z.is_floating_point()
        or reference_z.device != logits.device
        or reduction not in ("mean", "sum", "none")
    ):
        raise ValueError("Expected floating logits [rows,bins] and redshifts [rows] on one device")
    if selected is None:
        if len(logits) == 0:
            raise ValueError("Empty objective requires an explicit all-false selection mask")
        selected = torch.ones(len(logits), dtype=torch.bool, device=logits.device)
    elif (
        selected.shape != (len(logits),)
        or selected.dtype != torch.bool
        or selected.device != logits.device
    ):
        raise ValueError("selected must be a matching row-wise boolean mask")
    if not selected.any():
        empty = logits.reshape(-1)[:0]
        return empty if reduction == "none" else empty.sum()
    chosen = logits[selected].to(torch.float64)
    target = reference_z.detach()[selected].to(torch.float64)
    if (
        not torch.isfinite(chosen).all()
        or not torch.isfinite(target).all()
        or (target < edges[0]).any()
        or (target > edges[-1]).any()
    ):
        raise ValueError("Selected logits/truth must be finite and truth inside the declared grid")
    axis = torch.tensor(edges, dtype=torch.float64, device=logits.device)
    # Clamping here handles only the explicitly admitted last endpoint, not truth.
    index = (torch.bucketize(target, axis, right=True) - 1).clamp_max(len(edges) - 2)
    log_mass = F.log_softmax(chosen, dim=1)
    terms = -log_mass.gather(1, index[:, None]).squeeze(1) + torch.diff(axis)[index].log()
    if not torch.isfinite(terms).all():
        raise ValueError("Nonfinite density NLL; inspect logits and bin widths")
    return terms if reduction == "none" else terms.sum() if reduction == "sum" else terms.mean()


def _posterior_rows(
    bin_mass: ArrayLike, z_edges: ArrayLike
) -> tuple[FloatArray, FloatArray, BoolArray]:
    edges = _edges(z_edges)
    mass = _real_array(bin_mass, "bin_mass", 2)
    if mass.shape[1] != len(edges) - 1:
        raise ValueError("bin_mass must have one column per redshift bin")
    missing = np.isnan(mass).all(axis=1)
    available = np.asarray(np.isfinite(mass).all(axis=1), dtype=bool)
    if not (missing | available).all() or (mass[available] < 0).any():
        raise ValueError("Rows must be nonnegative finite bin masses or entirely NaN unsupported")
    with np.errstate(over="ignore", invalid="ignore"):
        totals = mass[available].sum(axis=1)
    if not np.allclose(totals, 1.0, rtol=1e-7, atol=1e-10):
        raise ValueError("Finite bin-mass rows must sum to one, not be unnormalized densities")
    # Correct only accepted floating-point roundoff, not arbitrary input weights.
    mass[available] /= totals[:, None]
    return mass, edges, available


def _probabilities(values: ArrayLike, name: str) -> FloatArray:
    probabilities = _real_array(values, name, 1)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(f"{name} must contain finite probabilities in [0,1]")
    return probabilities


def _quantiles(mass: FloatArray, edges: FloatArray, probabilities: FloatArray) -> FloatArray:
    result = np.full((len(mass), len(probabilities)), np.nan)
    if len(mass) == 0:
        return result
    cumulative = np.cumsum(mass, axis=1)
    cumulative[:, -1] = 1.0
    left_cumulative = np.concatenate([np.zeros((len(mass), 1)), cumulative[:, :-1]], axis=1)
    widths = np.diff(edges)
    rows = np.arange(len(mass))
    for column, probability in enumerate(probabilities):
        if probability == 0:
            index = (mass > 0).argmax(axis=1)
            result[:, column] = edges[index]
        elif probability == 1:
            index = mass.shape[1] - 1 - (mass[:, ::-1] > 0).argmax(axis=1)
            result[:, column] = edges[index + 1]
        else:
            # First CDF crossing gives the LEFT endpoint of a zero-mass plateau.
            index = (cumulative >= probability).argmax(axis=1)
            denominator = mass[rows, index]
            if (denominator <= 0).any():
                raise ValueError("Numerically unresolved quantile crossing")
            fraction = (probability - left_cumulative[rows, index]) / denominator
            result[:, column] = edges[index] + np.clip(fraction, 0, 1) * widths[index]
    return result


def posterior_quantiles(
    bin_mass: ArrayLike, z_edges: ArrayLike, probabilities: ArrayLike = (0.16, 0.5, 0.84)
) -> FloatArray:
    """Exact CDF interpolation; rows x probabilities, unsupported rows stay NaN.

    For q in (0,1), choose the first crossing, so an exact plateau boundary uses
    its leftmost endpoint. q=0/1 return first/last positive-mass support edges.
    No mode averaging or knot-as-point approximation is used.
    """
    mass, edges, available = _posterior_rows(bin_mass, z_edges)
    probabilities_array = _probabilities(probabilities, "probabilities")
    result = np.full((len(mass), len(probabilities_array)), np.nan)
    result[available] = _quantiles(mass[available], edges, probabilities_array)
    return result
