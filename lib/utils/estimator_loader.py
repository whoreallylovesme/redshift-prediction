from collections.abc import Callable, Mapping
from typing import Any

from sklearn.base import BaseEstimator

from lib.models.registry import FACTORIES, NN_FACTORIES, TABULAR_FACTORIES


def estimator_name(
    value: str, *, factories: Mapping[str, Callable[..., Any]] = FACTORIES
) -> str:
    if value not in factories:
        raise ValueError(f"Unknown estimator {value!r}; available: {sorted(factories)}")
    return value


def load_estimator(
    name: str,
    expected_type: type = BaseEstimator,
    *,
    params: dict[str, Any] | None = None,
    from_checkpoint: bool = False,
) -> object:
    factories: Mapping[str, Callable[..., Any]]
    if issubclass(expected_type, BaseEstimator):
        factories = TABULAR_FACTORIES
    else:
        from torch.nn import Module

        factories = NN_FACTORIES if issubclass(expected_type, Module) else FACTORIES
    factory = factories[estimator_name(name, factories=factories)]
    options = dict(params or {})
    if from_checkpoint and name in {"convnext_tiny", "efficientnet_b0", "vit_tiny"}:
        options["pretrained"] = False
    est = factory(**options)

    if not isinstance(est, expected_type):
        raise TypeError(f"Loaded object is not a {expected_type.__name__}")

    return est
