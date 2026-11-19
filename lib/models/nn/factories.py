from __future__ import annotations

from typing import Any


def convnext_tiny(image_channels: int = 3, pretrained: bool = True) -> Any:
    import timm

    model = timm.create_model(
        "convnext_tiny",
        pretrained=pretrained,
        in_chans=image_channels,
        num_classes=1,
        drop_rate=0.2,
    )

    model.requires_grad_(False)
    model.head.requires_grad_(True)

    return model


def efficientnet_b0(image_channels: int = 3, pretrained: bool = True) -> Any:
    import torch
    import torch.nn as nn
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    if image_channels < 1:
        raise ValueError("image_channels must be positive")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    if image_channels != 3:
        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        with torch.no_grad():
            if image_channels == 1:
                new_conv.weight[:, 0] = old_conv.weight.mean(dim=1)
            else:
                keep = min(3, image_channels)
                new_conv.weight[:, :keep] = old_conv.weight[:, :keep]
                if image_channels > 3:
                    new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        model.features[0][0] = new_conv

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )

    model.requires_grad_(False)
    model.classifier.requires_grad_(True)

    return model


def vit_tiny(image_channels: int = 3, pretrained: bool = True) -> Any:
    import timm

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        in_chans=image_channels,
        num_classes=1,
    )

    model.requires_grad_(False)
    model.head.requires_grad_(True)

    return model


def capsnet(
    image_channels: int = 3,
    input_size: int = 64,
    num_class: int = 2,
    dim_capsule: int = 16,
    routings: int = 3,
) -> Any:
    from lib.models.nn.capsnet import CapsuleRegressor

    return CapsuleRegressor(
        in_channels=image_channels,
        input_size=input_size,
        num_class=num_class,
        dim_capsule=dim_capsule,
        routings=routings,
    )


def fusion_cnn(image_channels: int, features: int, width: int = 32) -> Any:
    from lib.models.nn.fusion import FusionRegressor

    return FusionRegressor(image_channels, features, width)
