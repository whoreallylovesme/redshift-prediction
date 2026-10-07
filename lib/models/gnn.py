"""Трансдуктивная графовая сеть (GraphSAGE) для фото-z Euclid.

Узлы - объекты каталога; рёбра - k ближайших по небу с гауссовыми весами
по угловому расстоянию и разности цвета. Метки нужны только части узлов:
агрегация идёт по всему каталогу, лосс - по размеченным (semi-supervised).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SAGELayer(nn.Module):
    """Взвешенная агрегация признаков соседей + собственное преобразование."""

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.w_self = nn.Linear(dim_in, dim_out)
        self.w_nb = nn.Linear(dim_in, dim_out)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        deg: torch.Tensor,
    ) -> torch.Tensor:
        msg = self.w_nb(x[edge_index[0]]) * edge_weight[:, None]
        agg = torch.zeros(x.size(0), self.w_nb.out_features, device=x.device)
        agg.index_add_(0, edge_index[1], msg)
        agg = agg / deg.clamp(min=1.0)[:, None]
        return torch.relu(self.w_self(x) + agg)


class GraphSAGERegressor(nn.Module):
    """2-3 SAGE-слоя + голова; выход - логит, z = z_max * sigmoid(логит)."""

    def __init__(self, dim_in: int, hidden: int = 128, layers: int = 2, z_max: float = 3.0) -> None:
        super().__init__()
        self.z_max = z_max
        self.layers = nn.ModuleList(
            [SAGELayer(dim_in if i == 0 else hidden, hidden) for i in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        deg: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight, deg)
        return self.head(h).squeeze(-1)

    def predict_z(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        deg: torch.Tensor,
    ) -> torch.Tensor:
        """Красное смещение в [0, z_max] через сигмоиду от логита."""
        return self.z_max * torch.sigmoid(
            self.forward(x, edge_index, edge_weight, deg)
        )
