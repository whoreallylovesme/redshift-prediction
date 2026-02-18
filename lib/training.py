from typing import Any

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split

from .metrics import evaluate_regression


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
    split_cfg: dict[str, Any]
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.Series
]:
    X = df[features]
    y = df[target]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_seed"],
        shuffle=split_cfg["shuffle"],
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=split_cfg["val_size"],
        random_state=split_cfg["random_seed"],
        shuffle=split_cfg["shuffle"],
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, X_train_full, y_train_full


def hyperparameter_optimization(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cfg: dict[str, Any],
) -> Any:
    if not cfg["hyperopt"]["enabled"]:
        return {}

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=cfg["param_grid"],
        cv=cfg["hyperopt"]["cv"],
        scoring=cfg["hyperopt"]["scoring"],
        n_jobs=cfg["hyperopt"]["n_jobs"],
        verbose=cfg["hyperopt"]["verbose"],
    )
    gs.fit(X_train, y_train)

    return gs.best_params_


def train_model(
    estimator: Any,
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    best_params: dict[str, Any],
) -> Any:
    if best_params:
        estimator.set_params(**best_params)
    estimator.fit(X_train_full, y_train_full)

    return estimator


def evaluate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    pred = model.predict(X)

    return evaluate_regression(y, pred)
