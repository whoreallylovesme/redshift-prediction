from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lib.metrics.regression import Floats, mae, r2, rmse


def normalized_residual(truth: Floats, prediction: Floats) -> Floats:
    return (prediction - truth) / (1.0 + truth)


def delta(y_true: pd.Series | Floats, y_pred: Floats) -> Floats:
    return normalized_residual(
        np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    )


def bias(y_true: pd.Series | Floats, y_pred: Floats) -> float:
    return float(np.median(delta(y_true, y_pred)))


def nmad(y_true: pd.Series | Floats, y_pred: Floats) -> float:
    dz = delta(y_true, y_pred)
    return float(1.4826 * np.median(np.abs(dz - np.median(dz))))


def outlier_fraction(y_true: pd.Series | Floats, y_pred: Floats, threshold: float = 0.15) -> float:
    return float(np.mean(np.abs(delta(y_true, y_pred)) >= threshold))


def outlier_percent(y_true: pd.Series | Floats, y_pred: Floats, threshold: float = 0.15) -> float:
    return 100.0 * outlier_fraction(y_true, y_pred, threshold)


def photoz_stats(pred: Floats, z: Floats) -> tuple[float, float, float]:
    if not len(z):
        return np.nan, np.nan, np.nan
    return nmad(z, pred), outlier_fraction(z, pred), bias(z, pred)


def evaluate_regression(
    y_true: pd.Series | Floats, y_pred: NDArray[np.float64]
) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    coefficient = r2(truth, pred)
    return {
        "rmse": rmse(truth, pred),
        "mae": mae(truth, pred),
        "r2": coefficient if coefficient is not None else float("nan"),
        "bias": bias(y_true, y_pred),
        "sigma_nmad": nmad(y_true, y_pred),
        "f_outlier_percent": outlier_percent(y_true, y_pred),
    }
