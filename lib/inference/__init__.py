from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from runtime.inference import inference as inference

__all__ = ["inference", "build_predictor", "run_inference"]


def __getattr__(name: str) -> Any:
    if name == "inference":
        from runtime.inference import inference

        return inference
    if name in {"build_predictor", "run_inference"}:
        from lib.inference import engine

        return getattr(engine, name)
    raise AttributeError(name)
