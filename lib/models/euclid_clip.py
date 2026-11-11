from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from lib.utils.utils import check

FEATURE_DIM = 256
DESCRIPTOR_DIM = 18
TABLE_BANDS = 14
IMAGE_SIDE = 64
NBIN = 200
LMAX = float(np.log1p(6.0))
EDGES: NDArray[np.float64] = np.linspace(0.0, LMAX, NBIN + 1)
CENTRES: NDArray[np.float64] = 0.5 * (EDGES[1:] + EDGES[:-1])


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


def declared_mask(value: Tensor | None, batch: int, device: torch.device) -> Tensor:
    if value is None:
        return torch.ones(batch, dtype=torch.bool, device=device)
    check(
        value.shape == (batch,) and value.dtype == torch.bool and value.device == device,
        "availability must be boolean (batch,) on the input device",
    )
    return value


def boolean_mask(value: Tensor, shape: tuple[int, ...], device: torch.device) -> Tensor:
    check(
        value.shape == shape and value.dtype == torch.bool and value.device == device,
        "native validity must be boolean and match the input shape and device",
    )
    return value


def finite_observations(values: Tensor, valid: Tensor) -> Tensor:
    check(torch.isfinite(values[valid]).all(), "observed input contains NaN or Inf")
    return torch.where(valid, values, torch.zeros((), device=values.device, dtype=values.dtype))


@dataclass
class EncodedModality:
    features: Tensor
    available: Tensor


@dataclass
class EncodedIT:
    image: EncodedModality
    table: EncodedModality

    @property
    def available(self) -> Tensor:
        return self.image.available | self.table.available


def scatter_features(values: Tensor, indices: Tensor, batch: int) -> Tensor:
    return values.new_zeros((batch, FEATURE_DIM)).index_copy(0, indices, values)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.shortcut: nn.Module = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(h)) + self.shortcut(x))


class ImageEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(12, 32, 3, padding=1, bias=False), nn.GroupNorm(8, 32), nn.ReLU()
        )
        blocks: list[nn.Module] = []
        previous = 32
        for width in (32, 64, 128, 256):
            blocks += [
                ResidualBlock(previous, width, 1 if width == 32 else 2),
                ResidualBlock(width, width),
            ]
            previous = width
        self.blocks = nn.Sequential(*blocks)

    def forward(self, image: Tensor, image_available: Tensor | None = None) -> EncodedModality:
        check(
            image.ndim == 4 and image.shape[1:] == (12, IMAGE_SIDE, IMAGE_SIDE),
            "image must be Bx12x64x64",
        )
        batch = len(image)
        declared = declared_mask(image_available, batch, image.device)
        native = image[:, 8:12].detach()
        check(
            bool(torch.isfinite(native[declared]).all())
            and bool(torch.isin(native[declared], native.new_tensor([0.0, 1.0])).all()),
            "image pixel mask must be binary for a present image",
        )
        mask = (native > 0.5) & declared[:, None, None, None]
        available = mask.flatten(1).any(1)
        clean = finite_observations(image[:, :8], torch.cat((mask, mask), 1))
        clean = torch.cat((clean, mask.to(clean.dtype)), 1)
        active = torch.nonzero(available, as_tuple=False).flatten()
        if not len(active):
            return EncodedModality(image.new_zeros((batch, FEATURE_DIM)), available)
        h = self.blocks(self.stem(clean[active]))
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return EncodedModality(scatter_features(h, active, batch), available)


def attention_stack(layers: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=128,
        nhead=4,
        dim_feedforward=256,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class TableEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(2 + DESCRIPTOR_DIM, 128), nn.GELU(), nn.Linear(128, 128)
        )
        self.attention = attention_stack(2)
        self.norm = nn.LayerNorm(128)
        self.output = nn.Linear(128, FEATURE_DIM)

    def forward(
        self,
        table_values: Tensor,
        table_mask: Tensor,
        table_descriptor: Tensor,
        table_available: Tensor | None = None,
    ) -> EncodedModality:
        check(
            table_values.ndim == 3 and table_values.shape[1:] == (TABLE_BANDS, 2),
            "table values must be Bx14x2",
        )
        batch = len(table_values)
        valid = boolean_mask(table_mask, (batch, TABLE_BANDS), table_values.device)
        declared = declared_mask(table_available, batch, table_values.device)
        valid = valid & declared[:, None]
        available = valid.any(1)
        check(table_descriptor.device == table_values.device, "filter descriptors on input device")
        if table_descriptor.shape == (TABLE_BANDS, DESCRIPTOR_DIM):
            descriptor = table_descriptor.unsqueeze(0).expand(batch, -1, -1)
        else:
            check(
                table_descriptor.shape == (batch, TABLE_BANDS, DESCRIPTOR_DIM),
                "filter descriptor shape 14x18 or Bx14x18",
            )
            descriptor = table_descriptor
        check(torch.isfinite(descriptor).all(), "physical filter constants must all be finite")
        clean = finite_observations(table_values, valid[..., None].expand_as(table_values))
        active = torch.nonzero(available, as_tuple=False).flatten()
        if not len(active):
            return EncodedModality(table_values.new_zeros((batch, FEATURE_DIM)), available)
        mask = valid[active]
        tokens = self.token(torch.cat((clean[active], descriptor[active]), -1))
        encoded = self.norm(self.attention(tokens, src_key_padding_mask=~mask))
        pooled = (encoded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        features = self.output(pooled)
        return EncodedModality(scatter_features(features, active, batch), available)


class SpectrumEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch = nn.Linear(8 * 4, 128)
        self.wavelength = nn.Linear(1, 128, bias=False)
        self.attention = attention_stack(4)
        self.norm = nn.LayerNorm(128)
        self.metadata = nn.Sequential(nn.Linear(3, 32), nn.GELU())
        self.output = nn.Linear(128 + 32, FEATURE_DIM)

    def forward(
        self,
        spectrum: Tensor,
        spectrum_mask: Tensor,
        spectrum_metadata: Tensor,
        spectrum_available: Tensor | None = None,
    ) -> EncodedModality:
        check(
            spectrum.ndim == 3 and spectrum.shape[-1] == 4 and spectrum.shape[1] >= 1,
            "spectrum must be BxLx4",
        )
        batch, length, _ = spectrum.shape
        mask = boolean_mask(spectrum_mask, (batch, length), spectrum.device)
        declared = declared_mask(spectrum_available, batch, spectrum.device)
        mask = mask & declared[:, None]
        available = mask.any(1)
        check(
            spectrum_metadata.shape == (batch, 3) and spectrum_metadata.device == spectrum.device,
            "spectrum metadata Bx3",
        )
        metadata = finite_observations(
            spectrum_metadata, available[:, None].expand_as(spectrum_metadata)
        )
        observed = finite_observations(
            spectrum[..., [0, 1, 3]], mask[..., None].expand(batch, length, 3)
        )
        clean = torch.stack(
            (observed[..., 0], observed[..., 1], mask.to(observed.dtype), observed[..., 2]), -1
        )
        active = torch.nonzero(available, as_tuple=False).flatten()
        if not len(active):
            return EncodedModality(spectrum.new_zeros((batch, FEATURE_DIM)), available)
        clean, mask = clean[active], mask[active]
        pad = (-length) % 8
        if pad:
            clean = F.pad(clean, (0, 0, 0, pad))
            mask = F.pad(mask, (0, pad), value=False)
        n_patches = clean.shape[1] // 8
        chunks = clean.reshape(len(active), n_patches, 8, 4)
        bins = mask.reshape(len(active), n_patches, 8)
        patch_valid = bins.any(2)
        observed_bins = bins.sum(2, keepdim=True).clamp_min(1)
        centre = (chunks[..., 3] * bins).sum(2, keepdim=True) / observed_bins
        tokens = self.patch(chunks.flatten(2)) + self.wavelength(centre)
        encoded = self.norm(self.attention(tokens, src_key_padding_mask=~patch_valid))
        pooled = (encoded * patch_valid[..., None]).sum(1) / patch_valid.sum(1, keepdim=True)
        features = self.output(torch.cat((pooled, self.metadata(metadata[active])), 1))
        return EncodedModality(scatter_features(features, active, batch), available)


def photoz_head(inputs: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1)
    )


class ITNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.table_encoder = TableEncoder()
        self.photoz_head = photoz_head(2 * FEATURE_DIM + 2)

    def encode(
        self,
        *,
        image: Tensor,
        table_values: Tensor,
        table_mask: Tensor,
        table_descriptor: Tensor,
        image_available: Tensor | None = None,
        table_available: Tensor | None = None,
    ) -> EncodedIT:
        im: EncodedModality = self.image_encoder(image, image_available)
        tab: EncodedModality = self.table_encoder(
            table_values, table_mask, table_descriptor, table_available
        )
        check(len(im.features) == len(tab.features), "image and table batches aligned")
        return EncodedIT(im, tab)


class ProjectionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(FEATURE_DIM, FEATURE_DIM), nn.GELU(), nn.Linear(FEATURE_DIM, 128)
        )

    def forward(self, value: EncodedModality) -> Tensor:
        projected = F.normalize(self.net(value.features), dim=1, eps=1e-6)
        return torch.where(value.available[:, None], projected, torch.zeros_like(projected))


class CLIPModel(nn.Module):
    def __init__(self, it_network: ITNetwork | None = None, temperature: float = 0.1) -> None:
        super().__init__()
        check(math.isfinite(temperature) and temperature > 0, "positive fixed temperature")
        self.it_network = ITNetwork() if it_network is None else it_network
        self.spectrum_encoder = SpectrumEncoder()
        self.projectors = nn.ModuleDict({key: ProjectionHead() for key in ("I", "S", "T")})
        self.temperature = temperature

    def ssl_parameters(self) -> list[nn.Parameter]:
        modules: tuple[nn.Module, ...] = (
            self.it_network.image_encoder,
            self.it_network.table_encoder,
            self.spectrum_encoder,
            self.projectors,
        )
        return [parameter for module in modules for parameter in module.parameters()]

    def embeddings(
        self, batch: Mapping[str, Tensor]
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        fields = batch["field_code"].detach().cpu().tolist()
        check(
            len(fields) == len(batch["image"]) and all(f in (1, 2) for f in fields),
            "SSL only EDF-F/EDF-S",
        )
        encoded = self.it_network.encode(
            image=batch["image"],
            table_values=batch["table_values"],
            table_mask=batch["table_mask"],
            table_descriptor=batch["table_descriptor"],
            image_available=batch["image_available"],
            table_available=batch["table_available"],
        )
        spec: EncodedModality = self.spectrum_encoder(
            batch["spectrum"],
            batch["spectrum_mask"],
            batch["spectrum_metadata"],
            batch["spectrum_available"],
        )
        modalities = {"I": encoded.image, "S": spec, "T": encoded.table}
        embeddings = {key: self.projectors[key](value) for key, value in modalities.items()}
        available = {key: value.available for key, value in modalities.items()}
        return embeddings, available

    def forward(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, dict[str, int]]:
        embeddings, available = self.embeddings(batch)
        ids = batch["object_id"].detach().cpu().tolist()
        return masked_pairwise_clip(embeddings, available, ids, self.temperature)


class BinNet(nn.Module):
    def __init__(self, it: ITNetwork) -> None:
        super().__init__()
        self.it = it
        self.head = photoz_head(2 * FEATURE_DIM + 2)
        self.head[-1] = nn.Linear(128, NBIN)

    def forward(
        self,
        *,
        image: Tensor,
        table_values: Tensor,
        table_mask: Tensor,
        table_descriptor: Tensor,
        image_available: Tensor | None = None,
        table_available: Tensor | None = None,
    ) -> Tensor:
        enc = self.it.encode(
            image=image,
            table_values=table_values,
            table_mask=table_mask,
            table_descriptor=table_descriptor,
            image_available=image_available,
            table_available=table_available,
        )
        parts = [
            enc.image.features,
            enc.table.features,
            enc.image.available[:, None].float(),
            enc.table.available[:, None].float(),
        ]
        logits: Tensor = self.head(torch.cat(parts, 1))
        return logits


def soft_target(lz: NDArray[np.float64]) -> NDArray[np.float32]:
    x = np.clip((lz - CENTRES[0]) / (CENTRES[1] - CENTRES[0]), 0, NBIN - 1 - 1e-6)
    lo = np.floor(x).astype(int)
    w = x - lo
    t = np.zeros((len(lz), NBIN), np.float32)
    t[np.arange(len(lz)), lo] = 1 - w
    t[np.arange(len(lz)), lo + 1] = w
    return t


def bin_summary(p: NDArray[np.float32]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cdf = np.cumsum(p, 1)

    def quantile(level: float) -> NDArray[np.float64]:
        k = (cdf >= level).argmax(1)
        prev = np.where(k > 0, cdf[np.arange(len(p)), k - 1], 0.0)
        frac = (level - prev) / np.maximum(p[np.arange(len(p)), k], 1e-12)
        out: NDArray[np.float64] = EDGES[k] + frac * (EDGES[1] - EDGES[0])
        return out

    med, lo, hi = quantile(0.5), quantile(0.16), quantile(0.84)
    width: NDArray[np.float64] = np.log(np.maximum((hi - lo) / 2, 1e-3))
    return med, width
