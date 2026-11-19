from typing import Literal

import torch
from torch import Tensor, nn

from photoz.models.euclid_clip import (
    DESCRIPTOR_DIM,
    FEATURE_DIM,
    TABLE_BANDS,
    EncodedModality,
    TableEncoder,
    finite_observations,
    photoz_head,
)

type Architecture = Literal["resmlp", "table_transformer"]


class ResidualLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual: Tensor = self.net(x)
        return x + residual


class ResMLPEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(3 * TABLE_BANDS, 128)
        self.blocks = nn.Sequential(ResidualLayer(), ResidualLayer())
        self.output = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, FEATURE_DIM))

    def forward(
        self, values: Tensor, mask: Tensor, descriptor: Tensor, available: Tensor
    ) -> EncodedModality:
        if values.shape[1:] != (TABLE_BANDS, 2) or descriptor.shape != (
            TABLE_BANDS,
            DESCRIPTOR_DIM,
        ):
            raise ValueError(
                "Expected fourteen physical slots with two features and 18 descriptors"
            )
        observed = mask & available[:, None]
        clean = finite_observations(values, observed[..., None].expand_as(values))
        features = torch.cat((clean.flatten(1), observed.to(clean.dtype)), dim=1)
        features = self.output(self.blocks(nn.functional.gelu(self.input(features))))
        measured = observed.any(1)
        return EncodedModality(torch.where(measured[:, None], features, 0.0), measured)


class TableRegressor(nn.Module):
    def __init__(self, architecture: Architecture, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.table_encoder: nn.Module
        if architecture == "resmlp":
            self.table_encoder = ResMLPEncoder()
        elif architecture == "table_transformer":
            self.table_encoder = TableEncoder()
        else:
            raise ValueError(f"Unknown table architecture {architecture!r}")
        torch.manual_seed(seed + 5000)
        self.photoz_head = photoz_head(FEATURE_DIM + 1)

    def forward(self, values: Tensor, mask: Tensor, descriptor: Tensor) -> Tensor:
        encoded = self.table_encoder(values, mask, descriptor, mask.sum(1) >= 2)
        joined = torch.cat((encoded.features, encoded.available[:, None].to(values.dtype)), dim=1)
        prediction: Tensor = self.photoz_head(joined).squeeze(1)
        return torch.where(encoded.available, prediction, 0.0)
