from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from lib.metrics.regression import mae, r2, rmse

Floats = NDArray[np.float64]
Score = Callable[[Floats, Floats], float | None]
Direction = Literal["min", "max", "zero"]


@dataclass(frozen=True)
class Metric:
    label: str
    direction: Direction
    score: Score
    unit: str = ""


def residual(truth: Floats, prediction: Floats) -> Floats:
    return prediction - truth


def all_targets(truth: Floats) -> NDArray[np.bool_]:
    return np.isfinite(truth)


def diagonal(x: Floats) -> dict[str, Floats]:
    return {"Диагональ": x}


@dataclass(frozen=True)
class ReportTask:
    name: str
    metrics: Mapping[str, Metric]
    primary: str = "mae"
    target_label: str = "Эталон"
    prediction_label: str = "Прогноз"
    residual_label: str = "Прогноз − эталон"
    residual: Callable[[Floats, Floats], Floats] = residual
    valid_target: Callable[[Floats], NDArray[np.bool_]] = all_targets
    guides: Callable[[Floats], dict[str, Floats]] = diagonal


REGRESSION = ReportTask(
    name="regression",
    metrics={
        "mae": Metric("MAE", "min", mae),
        "rmse": Metric("RMSE", "min", rmse),
        "r2": Metric("R²", "max", r2),
    },
)


def resolve_task(task: str | ReportTask) -> ReportTask:
    if isinstance(task, ReportTask):
        return task
    if task == REGRESSION.name:
        return REGRESSION
    raise ValueError(f"Pass a configured ReportTask for {task!r}")


def scores(task: ReportTask, truth: Floats, pred: Floats) -> dict[str, float | None]:
    return {
        key: metric.score(truth, pred) if len(truth) else None
        for key, metric in task.metrics.items()
    }
