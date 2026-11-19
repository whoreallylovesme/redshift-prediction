from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import NDArray

from lib.metrics.regression import mae, r2, rmse

type MetricFn = Callable[[pd.Series, NDArray[np.float64]], dict[str, float]]


def regression_metrics(truth: pd.Series, prediction: NDArray[np.float64]) -> dict[str, float]:
    observed = np.asarray(truth, dtype=np.float64)
    coeff = r2(observed, prediction)
    return {
        "mae": mae(observed, prediction),
        "rmse": rmse(observed, prediction),
        "r2": coeff if coeff is not None else float("nan"),
    }


@dataclass
class RegressionTask:
    criterion: nn.Module = field(default_factory=nn.MSELoss)
    epoch_metrics: MetricFn = regression_metrics
    monitor: str = "val/rmse"
    mode: Literal["min", "max"] = "min"


def scalar_predictions(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if value.shape not in {(batch_size,), (batch_size, 1)}:
        raise ValueError("Regressor must return one scalar per observation")
    return value.reshape(batch_size)
