from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import bootstrap

type Indices = NDArray[np.intp]


def group_members(labels: pd.Series | NDArray[Any]) -> list[Indices]:
    values = pd.Series(labels, copy=False)
    return list(values.groupby(values, sort=True, dropna=False, observed=True).indices.values())


def bootstrap_distribution(
    statistic: Callable[..., float | NDArray[Any]], sizes: tuple[int, ...], repeats: int, seed: int
) -> NDArray[Any]:
    """Resample indices on independent axes, with the replicate on the last axis."""
    if not sizes or min(sizes) < 1 or repeats < 1:
        raise ValueError("Bootstrap requires nonempty samples and at least one resample")
    units = tuple(np.arange(size, dtype=np.intp) for size in sizes)
    # SciPy requires two or more units per sample. A singleton is fixed in every draw.
    variable = tuple(axis for axis in units if len(axis) > 1)
    if not variable:
        fixed = np.asarray(statistic(*units))
        return np.repeat(fixed[..., None], repeats, axis=-1)

    def evaluate(*draws: Indices) -> float | NDArray[Any]:
        sampled = iter(draws)
        return statistic(*(next(sampled) if len(axis) > 1 else axis for axis in units))

    # Draw all axes of one replicate together to preserve the seeded Euclid protocol.
    result = bootstrap(
        variable, evaluate, n_resamples=repeats, method="percentile", vectorized=False,
        batch=1, rng=np.random.default_rng(seed),
    )
    return np.asarray(result.bootstrap_distribution)


_EUCLID_NAMES = [
    "ARM_SEED",
    "MIN_ROWS",
    "N_BOOT",
    "OUTLIER_THRESHOLD",
    "SEED_TILE_SEED",
    "TILE_SEED",
    "FloatArray",
    "Record",
    "arm_boot",
    "arm_summary",
    "metric_record",
    "photoz_stats",
    "seed_summary",
    "seed_tile_boot",
    "stats_record",
    "tile_boot",
    "tile_members",
]
__all__ = ["Indices", "bootstrap_distribution", "group_members", *_EUCLID_NAMES]


def __getattr__(name: str) -> Any:
    if name in _EUCLID_NAMES:
        from photoz import bootstrap as euclid

        return getattr(euclid, name)
    raise AttributeError(name)
