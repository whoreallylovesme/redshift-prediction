from typing import Any

from sklearn.base import BaseEstimator

from lib.models.factories import FACTORIES


def estimator_name(value: str) -> str:
    # Historical trusted recipes also supply a Python snippet that constructs `estimator`.
    if value not in FACTORIES and "\n" not in value and "estimator =" not in value:
        raise ValueError(f"Unknown estimator {value!r}; available: {sorted(FACTORIES)}")
    return value


def load_estimator(
    estimator_code: str,
    expected_type: type = BaseEstimator,
    *,
    params: dict[str, Any] | None = None,
    from_checkpoint: bool = False,
) -> object:
    factory = FACTORIES.get(estimator_name(estimator_code))
    if factory is not None:
        options = dict(params or {})
        if from_checkpoint and estimator_code in {"convnext_tiny", "efficientnet_b0", "vit_tiny"}:
            options["pretrained"] = False
        est = factory(**options)
    else:
        if params:
            raise ValueError("Constructor parameters require a named estimator factory")
        scope: dict[str, object] = {}
        exec(estimator_code, scope)
        est = scope["estimator"]

    if not isinstance(est, expected_type):
        raise TypeError(f"Loaded object is not a {expected_type.__name__}")

    return est
