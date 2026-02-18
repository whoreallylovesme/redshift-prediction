import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {"rmse": rmse, "mae": mae, "r2": r2}