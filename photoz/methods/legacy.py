from pathlib import Path
from typing import Any


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


NAMED_PREDICTORS = {"capsnet": predict_capsnet, "euclid_clip": predict_clip}
