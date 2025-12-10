import pandas as pd
from sklearn.model_selection import GridSearchCV

from .config import ExperimentConfig
from .models import make_pipeline
from .metrics import evaluate_regression


def run_experiments(
    configs,
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
):
    rows = []
    trained_models = {}

    for cfg in configs:
        print(f"Train model >> {cfg.name} ({cfg.model_type}) <<")

        pipe = make_pipeline(cfg)
        pipe.fit(X_train, y_train)

        y_val_pred = pipe.predict(X_val)
        val_metrics = evaluate_regression(y_val, y_val_pred)

        y_test_pred = pipe.predict(X_test)
        test_metrics = evaluate_regression(y_test, y_test_pred)

        rows.append({
            "name": cfg.name,
            "model_type": cfg.model_type,
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_r2": val_metrics["r2"],
            "test_rmse": test_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_r2": test_metrics["r2"],
        })

        trained_models[cfg.name] = pipe

    results_df = pd.DataFrame(rows).sort_values("val_rmse")
    return results_df, trained_models


def run_grid_search(
    base_cfg: ExperimentConfig,
    param_grid,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=0,
):
    pipe = make_pipeline(base_cfg)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    grid.fit(X_train, y_train)
    return grid


def extract_model_params(best_params):
    model_params = {}
    for key, value in best_params.items():
        if key.startswith("model__"):
            clean_key = key.split("model__")[1]
            model_params[clean_key] = value
    return model_params
