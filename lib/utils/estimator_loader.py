from typing import Any

from sklearn.base import BaseEstimator

from lib.models.registry import FACTORIES


def estimator_name(value: str) -> str:
    if value not in FACTORIES:
        raise ValueError(f"Unknown estimator {value!r}; available: {sorted(FACTORIES)}")
    return value


def load_estimator(
    name: str,
    expected_type: type = BaseEstimator,
    *,
    params: dict[str, Any] | None = None,
    from_checkpoint: bool = False,
) -> object:
    factory = FACTORIES[estimator_name(name)]
    options = dict(params or {})
    if from_checkpoint and name in {"convnext_tiny", "efficientnet_b0", "vit_tiny"}:
        options["pretrained"] = False
    est = factory(**options)

    if not isinstance(est, expected_type):
        raise TypeError(f"Loaded object is not a {expected_type.__name__}")

    return est
