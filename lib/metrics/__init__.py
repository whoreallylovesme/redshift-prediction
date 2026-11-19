from importlib import import_module
from typing import Any

from .regression import mae, r2, rmse

_BOOTSTRAP = {
    "arm_summary",
    "metric_record",
    "seed_summary",
    "seed_tile_boot",
    "stats_record",
    "tile_boot",
}
_LEGACY = {
    "bias",
    "delta",
    "evaluate_regression",
    "nmad",
    "outlier_fraction",
    "outlier_percent",
    "photoz_stats",
}
__all__ = ["mae", "r2", "rmse", *sorted(_BOOTSTRAP | _LEGACY)]


def __getattr__(name: str) -> Any:
    if name in _BOOTSTRAP:
        return getattr(import_module("photoz.bootstrap"), name)
    if name in _LEGACY:
        return getattr(import_module("photoz.metrics"), name)
    raise AttributeError(name)
