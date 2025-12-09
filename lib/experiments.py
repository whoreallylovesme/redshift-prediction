import pandas as pd

from config import ExperimentConfig
from models import make_pipeline
from metrics import evaluate_regression


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
        val_metrics = regression_metrics(y_val, y_val_pred)

        y_test_pred = pipe.predict(X_test)
        test_metrics = regression_metrics(y_test, y_test_pred)

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
