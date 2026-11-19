import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch import nn

from lib.config import PytorchTrainConfig
from lib.train.regression import RegressionTask
from lib.train.utils import WeightedHuberLoss
from photoz.metrics import bias, evaluate_regression, nmad, outlier_percent


def epoch_metrics(truth: pd.Series, prediction: NDArray[np.float64]) -> dict[str, float]:
    return {
        "nmad": nmad(truth, prediction),
        "bias": bias(truth, prediction),
        "outlier_pct": outlier_percent(truth, prediction),
    }


def photoz_regression_task(cfg: PytorchTrainConfig) -> RegressionTask:
    criterion: nn.Module = nn.HuberLoss(delta=0.1)
    if cfg.z_weight_alpha > 0:
        criterion = WeightedHuberLoss(delta=0.1, alpha=cfg.z_weight_alpha)
    return RegressionTask(criterion=criterion, epoch_metrics=epoch_metrics, monitor="val/nmad")


sklearn_metrics = evaluate_regression
