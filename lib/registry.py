from collections.abc import Callable
from typing import Any

Trainer = Callable[..., Any]
Predictor = Callable[..., Any]

TORCH_TRAINERS: dict[str, Trainer] = {}
TORCH_PREDICTORS: dict[str, Predictor] = {}


def register_torch_trainer(model: str, fn: Trainer) -> None:
    TORCH_TRAINERS[model] = fn


def register_torch_predictor(model: str, fn: Predictor) -> None:
    TORCH_PREDICTORS[model] = fn
