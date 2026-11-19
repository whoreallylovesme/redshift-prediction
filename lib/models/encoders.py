"""Reusable residual and attention layers for images, feature vectors and token sequences."""

from torch import Tensor, nn
from torch.nn import functional as F


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
