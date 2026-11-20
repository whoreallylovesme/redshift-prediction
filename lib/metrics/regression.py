import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

Floats = NDArray[np.float64]


def mae(truth: Floats, prediction: Floats) -> float:
    return float(mean_absolute_error(truth, prediction))


def rmse(truth: Floats, prediction: Floats) -> float:
    return float(np.sqrt(mean_squared_error(truth, prediction)))


def r2(truth: Floats, prediction: Floats) -> float | None:
    return float(r2_score(truth, prediction)) if len(truth) > 1 else None
