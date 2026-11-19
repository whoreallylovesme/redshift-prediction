from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from lib.config import SklearnTrainConfig
from lib.train.regression import MetricFn, regression_metrics


def feature_selection(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    cols = set(df.columns)
    selected = []
    missing = []
    for f in features:
        if f in cols:
            selected.append(f)
        else:
            missing.append(f)

    return selected, missing


def split_data(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    test_size: float,
    val_size: float,
    shuffle: bool,
    random_seed: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    X = df[features]
    y = df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        shuffle=shuffle,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=random_seed,
        shuffle=shuffle,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_full, y_train_full


def hyperparameter_optimization(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: SklearnTrainConfig,
    weights: pd.Series | None = None,
) -> Any:
    if not cfg.hyperopt_enabled:
        return {}

    from sklearn.metrics import get_scorer

    metric = get_scorer(cfg.hyperopt_scoring)

    def weighted_score(model: Any, x: pd.DataFrame, y: pd.Series) -> float:
        assert weights is not None
        return float(metric(model, x, y, sample_weight=weights.loc[x.index].to_numpy()))

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=cfg.param_grid,
        cv=cfg.hyperopt_cv,
        scoring=cfg.hyperopt_scoring if weights is None else weighted_score,
        n_jobs=cfg.hyperopt_n_jobs,
        verbose=cfg.hyperopt_verbose,
    )
    gs.fit(X_train, y_train, **weight_parameters(estimator, weights))

    return gs.best_params_


def weight_parameters(estimator: Any, weights: pd.Series | None) -> dict[str, Any]:
    if weights is None:
        return {}
    from sklearn.pipeline import Pipeline
    from sklearn.utils.validation import has_fit_parameter

    key, final = "sample_weight", estimator
    if isinstance(estimator, Pipeline):
        name, final = estimator.steps[-1]
        key = f"{name}__sample_weight"
    if not has_fit_parameter(final, "sample_weight"):
        raise ValueError(f"{type(final).__name__} cannot train with configured sample weights")
    return {key: weights.to_numpy()}


def train_model(
    estimator: Any,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    best_params: dict[str, Any],
    weights: pd.Series | None = None,
) -> Any:
    if best_params:
        estimator.set_params(**best_params)
    estimator.fit(X_train_full, y_train_full, **weight_parameters(estimator, weights))

    return estimator


def evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    metrics: MetricFn = regression_metrics,
) -> dict[str, float]:
    pred = model.predict(X)
    result: dict[str, float] = metrics(y, pred)
    return result


def fit_regressor(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    cfg: SklearnTrainConfig,
    *,
    threads: int | None,
    metrics: Callable[..., Any],
    weight_column: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    from lib.utils.estimator_loader import load_estimator

    x, y = frame[features], frame[target]
    if cfg.val_size:
        xt, xv, yt, yv = train_test_split(
            x, y, test_size=cfg.val_size, random_state=cfg.random_seed, shuffle=cfg.shuffle
        )
    else:
        xt, xv, yt, yv = x, x.iloc[:0], y, y.iloc[:0]
    estimator: Any = load_estimator(cfg.estimator)
    overrides = dict(cfg.estimator_params)
    if threads is not None:
        for name in ("model__n_jobs", "model__thread_count"):
            if name in estimator.get_params():
                overrides[name] = threads
    estimator.set_params(**overrides)
    weights = frame.loc[xt.index, weight_column] if weight_column else None
    best = hyperparameter_optimization(estimator, xt, yt, cfg, weights)
    model = train_model(estimator, xt, yt, best, weights)
    return model, {
        "selected_features": features,
        "best_params": best,
        "train_rows": len(yt),
        "validation_rows": len(yv),
        "val_metrics": metrics(yv, model.predict(xv)) if len(yv) else {},
        "weight_column": weight_column,
    }
