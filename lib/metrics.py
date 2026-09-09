import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def delta(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    y_true_arr = y_true.to_numpy(dtype=np.float64)
    y_pred_arr = y_pred.astype(np.float64)
    out: NDArray[np.float64] = (y_pred_arr - y_true_arr) / (1.0 + y_true_arr)
    return out


def bias(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
) -> float:
    dz = delta(y_true, y_pred)
    return float(np.median(dz))


def nmad(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
) -> float:
    dz = delta(y_true, y_pred)
    bias = np.median(dz)
    return float(1.4826 * np.median(np.abs(dz - bias)))


def outlier_fraction(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
    threshold: float = 0.15,
) -> float:
    dz = delta(y_true, y_pred)
    return float(np.mean(np.abs(dz) >= threshold))


def outlier_percent(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
    threshold: float = 0.15,
) -> float:
    return 100.0 * outlier_fraction(y_true, y_pred, threshold)


def evaluate_regression(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "f_outlier_percent": outlier_percent(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "sigma_nmad": nmad(y_true, y_pred)
    }
