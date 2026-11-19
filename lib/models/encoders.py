"""Reusable residual and attention layers for images, feature vectors and token sequences."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lib.utils.utils import check


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stride: int = 1, *, groups: int = 8
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.shortcut: nn.Module = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        h = F.relu(self.norm1(self.conv1(x)))
        return F.relu(self.norm2(self.conv2(h)) + self.shortcut(x))


def attention_stack(
    layers: int, *, width: int = 128, heads: int = 4, dropout: float = 0.0, norm_first: bool = True
) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=2 * width,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=norm_first,
    )
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class ResidualLayer(nn.Module):
    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual: Tensor = self.net(x)
        return x + residual


class ResidualMLPFeatures(nn.Module):
    """Map a fixed-length feature vector to an embedding or regression output."""

    def __init__(
        self, in_features: int, out_features: int, *, width: int = 128, depth: int = 2
    ) -> None:
        super().__init__()
        if min(in_features, out_features, width, depth) < 1:
            raise ValueError("ResidualMLP dimensions and depth must be positive")
        self.input = nn.Linear(in_features, width)
        self.blocks = nn.Sequential(*(ResidualLayer(width) for _ in range(depth)))
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, out_features))

    def encode(self, features: Tensor) -> Tensor:
        out: Tensor = self.output(self.blocks(F.gelu(self.input(features))))
        return out


class ResidualMLP(ResidualMLPFeatures):
    def forward(self, features: Tensor) -> Tensor:
        return self.encode(features)


def regression_head(
    inputs: int, hidden: tuple[int, int] = (256, 128), outputs: int = 1
) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(inputs, hidden[0]),
        nn.GELU(),
        nn.Linear(*hidden),
        nn.GELU(),
        nn.Linear(hidden[1], outputs),
    )


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


class MaskedTableEncoder(nn.Module):
    def __init__(
        self, slots: int, value_dim: int, descriptor_dim: int, output_dim: int, *,
        width: int = 128, heads: int = 4, layers: int = 2,
    ) -> None:
        super().__init__()
        if min(slots, value_dim, descriptor_dim, output_dim, width, heads, layers) < 1:
            raise ValueError("Table encoder dimensions must be positive")
        self.input_shape = (slots, value_dim, descriptor_dim)
        self.token = nn.Sequential(
            nn.Linear(value_dim + descriptor_dim, width), nn.GELU(), nn.Linear(width, width)
        )
        self.attention = attention_stack(layers, width=width, heads=heads)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_dim)

    def forward(
        self,
        table_values: Tensor,
        table_mask: Tensor,
        table_descriptor: Tensor,
        table_available: Tensor | None = None,
    ) -> EncodedModality:
        slots, value_dim, descriptor_dim = self.input_shape
        check(
            table_values.ndim == 3 and table_values.shape[1:] == (slots, value_dim),
            f"Expected table values with shape [batch, {slots}, {value_dim}]",
        )
        batch = len(table_values)
        valid = boolean_mask(table_mask, (batch, slots), table_values.device)
        declared = declared_mask(table_available, batch, table_values.device)
        valid = valid & declared[:, None]
        available = valid.any(1)
        check(table_descriptor.device == table_values.device, "Descriptors on input device")
        if table_descriptor.shape == (slots, descriptor_dim):
            descriptor = table_descriptor.unsqueeze(0).expand(batch, -1, -1)
        else:
            check(
                table_descriptor.shape == (batch, slots, descriptor_dim),
                "Expected one descriptor per slot, shared or per batch row",
            )
            descriptor = table_descriptor
        check(torch.isfinite(descriptor).all(), "Descriptors must be finite")
        clean = finite_observations(table_values, valid[..., None].expand_as(table_values))
        active = torch.nonzero(available, as_tuple=False).flatten()
        if not len(active):
            output = table_values.new_zeros(
                (batch, self.output.out_features), dtype=self.output.weight.dtype
            )
            return EncodedModality(output, available)
        mask = valid[active]
        tokens = self.token(torch.cat((clean[active], descriptor[active]), -1))
        encoded = self.norm(self.attention(tokens, src_key_padding_mask=~mask))
        pooled = (encoded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        features = self.output(pooled)
        output = features.new_zeros((batch, features.shape[1])).index_copy(0, active, features)
        return EncodedModality(output, available)
