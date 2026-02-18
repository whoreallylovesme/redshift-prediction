from typing import Any

from sklearn.base import BaseEstimator


def load_estimator(cfg: dict[str, Any]) -> BaseEstimator:
    scope: dict[str, Any] = {}
    exec(cfg["estimator"], scope)

    est = scope["estimator"]

    if not isinstance(est, BaseEstimator):
        raise TypeError("Loaded object is not a BaseEstimator")

    return est
