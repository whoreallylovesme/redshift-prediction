import math
from collections.abc import Mapping, Sequence
from itertools import combinations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lib.models.encoders import EncodedModality, boolean_mask


def masked_pairwise_clip(
    embeddings: Mapping[str, Tensor],
    available: Mapping[str, Tensor],
    object_ids: Sequence[int],
    temperature: float = 0.1,
) -> tuple[Tensor, dict[str, int]]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if len(embeddings) < 2 or set(embeddings) != set(available):
        raise ValueError("need matching embedding and mask keys for at least two modalities")
    ids = list(object_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("batch ids must be nonempty and unique")
    expected: tuple[int, torch.device, torch.dtype] | None = None
    for name, z in embeddings.items():
        mask = available[name]
        if z.ndim != 2 or z.shape[0] != len(ids) or z.shape[1] < 1:
            raise ValueError("each embedding must have shape (batch, shared_dim)")
        signature = (z.shape[1], z.device, z.dtype)
        if not z.is_floating_point() or (expected is not None and signature != expected):
            raise ValueError("embeddings need matching floating dtype, device and dimension")
        expected = signature
        if mask.shape != (len(ids),) or mask.dtype != torch.bool or mask.device != z.device:
            raise ValueError("availability must be a boolean batch mask on the same device")
        measured = z[mask]
        if not torch.isfinite(measured).all() or (measured.norm(dim=1) == 0).any():
            raise ValueError("available embeddings must be finite and nonzero")
    losses: list[Tensor] = []
    counts: dict[str, int] = {}
    for a, b in combinations(sorted(embeddings), 2):
        overlap = available[a] & available[b]
        n = int(overlap.sum().item())
        counts[f"{a}__{b}"] = n
        if n < 2:
            continue
        za = F.normalize(embeddings[a][overlap], dim=1)
        zb = F.normalize(embeddings[b][overlap], dim=1)
        logits = za @ zb.T / temperature
        targets = torch.arange(n, device=logits.device)
        losses.append((F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2)
    if not losses:
        raise ValueError("no modality pair has two distinct observed objects")
    return torch.stack(losses).mean(), counts


class ProjectionHead(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features), nn.GELU(), nn.Linear(in_features, out_features)
        )

    def forward(self, value: EncodedModality) -> Tensor:
        features = value.features
        if features.ndim != 2:
            raise ValueError("Projection input must be a matrix of feature vectors")
        available = boolean_mask(value.available, (len(features),), features.device)[:, None]
        clean = torch.where(available, features, 0.0)
        projected = F.normalize(self.net(clean), dim=1, eps=1e-6)
        return torch.where(available, projected, torch.zeros_like(projected))
