"""The native redshift axis and historical density-head checkpoint schema."""

from __future__ import annotations

from typing import Any, Literal

import torch
from numpy.typing import ArrayLike

from lib.models import density
from lib.models.density import BinnedDensityHead, FloatArray
from lib.models.density import DensityPrediction as DensityPrediction


def _redshift_edges(values: ArrayLike) -> FloatArray:
    edges = density.bin_edges(values)
    if edges[0] <= -1:
        raise ValueError("Physical redshift edges must start above -1")
    return edges


class PhotoZDensityHead(BinnedDensityHead):
    def __init__(
        self, embedding_dim: int, z_edges: ArrayLike, *, hidden_dim: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__(
            embedding_dim, _redshift_edges(z_edges), hidden_dim=hidden_dim, dropout=dropout
        )

    @property
    def z_edges(self) -> FloatArray:
        return self.edges

    def config_dict(self) -> dict[str, Any]:
        result = super().config_dict()
        result["z_edges"] = result.pop("edges")
        result["output"] = "categorical bin mass; piecewise-uniform density in physical z"
        return result


def piecewise_uniform_nll(
    logits: torch.Tensor,
    reference_z: torch.Tensor,
    z_edges: ArrayLike,
    *,
    selected: torch.Tensor | None = None,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    return density.piecewise_uniform_nll(
        logits, reference_z, _redshift_edges(z_edges), selected=selected, reduction=reduction
    )


def posterior_quantiles(
    bin_mass: ArrayLike, z_edges: ArrayLike, probabilities: ArrayLike = (0.16, 0.5, 0.84)
) -> FloatArray:
    return density.posterior_quantiles(bin_mass, _redshift_edges(z_edges), probabilities)
