from __future__ import annotations

from typing import Any


def catboost() -> Any:
    from catboost import CatBoostRegressor
    from sklearn.pipeline import Pipeline

    estimator = Pipeline(
        [
            (
                "model",
                CatBoostRegressor(
                    loss_function="RMSE",
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=-1,
                ),
            )
        ]
    )
    return estimator


def imputed_catboost() -> Any:
    from sklearn.impute import SimpleImputer

    estimator = catboost()
    estimator.steps.insert(0, ("imputer", SimpleImputer(strategy="median")))
    return estimator


def gam() -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from lib.models.tabular.gam import GAMRegressor

    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GAMRegressor(n_splines=20, lam=0.6, max_iter=1000)),
        ]
    )
    return estimator


def llr() -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from lib.models.tabular.llr import LocalLinearRegression

    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LocalLinearRegression(k=1200)),
        ]
    )
    return estimator


def mlp() -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(32, 16, 8),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    learning_rate_init=1e-3,
                    max_iter=5000,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=30,
                    random_state=42,
                ),
            ),
        ]
    )
    return estimator


def rf() -> Any:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    estimator = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )
    return estimator


def ridge() -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("model", Ridge())]
    )
