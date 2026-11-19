import torch
from torch import Tensor, nn


class FusionRegressor(nn.Module):
    def __init__(self, image_channels: int, features: int, width: int = 32) -> None:
        super().__init__()
        if min(image_channels, features, width) < 1:
            raise ValueError("Image channels, feature count and width must be positive")
        self.image = nn.Sequential(
            nn.Conv2d(image_channels, width, 3, padding=1), nn.GELU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.table = nn.Sequential(nn.Linear(features, width), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, inputs: dict[str, Tensor]) -> Tensor:
        encoded = torch.cat((self.image(inputs["image"]), self.table(inputs["features"])), dim=1)
        prediction: Tensor = self.head(encoded)
        return prediction
