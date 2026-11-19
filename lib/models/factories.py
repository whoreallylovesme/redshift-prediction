from __future__ import annotations

from collections.abc import Callable
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

    for name, param in model.named_parameters():
        if not name.startswith("head"):
            param.requires_grad = False

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

    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False

    return model


def vit_tiny(image_channels: int = 3, pretrained: bool = True) -> Any:
    import timm

    model = timm.create_model(
        "vit_tiny_patch16_224",
        pretrained=pretrained,
        in_chans=image_channels,
        num_classes=1,
    )

    for name, param in model.named_parameters():
        if not name.startswith("head"):
            param.requires_grad = False

    return model


def catboost() -> Any:
    from catboost import CatBoostRegressor
    from sklearn.pipeline import Pipeline

    estimator = Pipeline(
        [
            (
                "model",
                CatBoostRegressor(
                    loss_function="RMSE",
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                ),
            )
        ]
    )
    return estimator


def imputed_catboost() -> Any:
    from sklearn.impute import SimpleImputer

    estimator = catboost()
    estimator.steps.insert(0, ("imputer", SimpleImputer(strategy="median")))
    return estimator


def gam() -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from lib.models.gam import GAMRegressor

    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GAMRegressor(n_splines=20, lam=0.6, max_iter=1000)),
        ]
    )
    return estimator


def llr() -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from lib.models.llr import LocalLinearRegression

    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LocalLinearRegression(k=1200)),
        ]
    )
    return estimator


def mlp() -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(32, 16, 8),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=5000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=30,
                    random_state=42,
                ),
            ),
        ]
    )
    return estimator


def rf() -> Any:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )
    return estimator


def ridge() -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", Ridge())]
    )


def fusion_cnn(image_channels: int, features: int, width: int = 32) -> Any:
    from lib.models.fusion import FusionRegressor

    return FusionRegressor(image_channels, features, width)


FACTORIES: dict[str, Callable[..., Any]] = {
    "fusion_cnn": fusion_cnn,
    "convnext_tiny": convnext_tiny,
    "efficientnet_b0": efficientnet_b0,
    "vit_tiny": vit_tiny,
    "catboost": catboost,
    "imputed_catboost": imputed_catboost,
    "gam": gam,
    "llr": llr,
    "mlp": mlp,
    "rf": rf,
    "ridge": ridge,
}
