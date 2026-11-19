from typing import Any

from lib.inference.regression import inference as inference

__all__ = ["inference", "build_predictor", "run_inference"]


def __getattr__(name: str) -> Any:
    if name in {"build_predictor", "run_inference"}:
        from lib.inference import engine

        return getattr(engine, name)
    raise AttributeError(name)
