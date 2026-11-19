import numpy as np
from numpy.typing import NDArray
from scipy.stats import median_abs_deviation


def scaled_mad(values: NDArray[np.floating], *, median: float | None = None) -> float:
    center = np.median if median is None else lambda x, axis: median
    return float(1.4826 * median_abs_deviation(values, axis=None, center=center))
