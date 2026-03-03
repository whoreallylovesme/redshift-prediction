import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def photoz_dz(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
) -> NDArray[np.float64]:
    y_true_arr = y_true.to_numpy(dtype=np.float64)
    y_pred_arr = y_pred.astype(np.float64)
    out: NDArray[np.float64] = (y_pred_arr - y_true_arr) / (1.0 + y_true_arr)
    return out


def photoz_bias(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
) -> float:
    dz = photoz_dz(y_true, y_pred)
    return float(np.median(dz))


def photoz_nmad(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
) -> float:
    dz = photoz_dz(y_true, y_pred)
    bias = np.median(dz)
    return float(1.4826 * np.median(np.abs(dz - bias)))


def photoz_outlier_fraction(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
    threshold: float = 0.15,
) -> float:
    dz = photoz_dz(y_true, y_pred)
    return float(np.mean(np.abs(dz) >= threshold))


def photoz_outlier_percent(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
    threshold: float = 0.15,
) -> float:
    return 100.0 * photoz_outlier_fraction(y_true, y_pred, threshold)


def evaluate_regression(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "f_outlier_percent": photoz_outlier_percent(y_true, y_pred),
        "bias": photoz_bias(y_true, y_pred),
        "sigma_nmad": photoz_nmad(y_true, y_pred)
    }
