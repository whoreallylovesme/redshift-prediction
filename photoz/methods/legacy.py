from pathlib import Path
from typing import Any

from lib.inference.regression import predict_images
from lib.utils.registry import register_torch_predictor


def build_predictor(raw: dict[str, Any], weights: Path) -> Any:
    from photoz.methods.capsnet import build_predictor as build

    return build(raw, weights)


def run_inference(config_path: Path) -> None:
    from photoz.methods.capsnet import run_inference as run

    run(config_path)


def predict_capsnet(config_path: str, **_: object) -> None:
    run_inference(Path(config_path))


def predict_clip(config_path: str, **_: object) -> None:
    from photoz.methods.euclid_clip import run_inference as run

    run(Path(config_path))


def register_predictors() -> None:
    register_torch_predictor("", predict_images)
    register_torch_predictor("capsnet", predict_capsnet)
    register_torch_predictor("euclid_clip", predict_clip)
