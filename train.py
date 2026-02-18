import json
from pathlib import Path
from typing import Any

import click
import joblib
import yaml

from lib.dataset import clean_dataset, read_dataset
from lib.estimator_loader import load_estimator
from lib.plots import redshift_scatter
from lib.training import (
    evaluate,
    feature_selection,
    hyperparameter_optimization,
    split_data,
    train_model,
)


def load_yaml(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--dataset", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--features", "features_path", required=True, type=click.Path(exists=True))
@click.option("--feature-set", "feature_set", default=None)
@click.option("--outdir", default="runs/last")
def main(
    dataset: str,
    config_path: str,
    features_path: str,
    feature_set: str | None,
    outdir: Path
) -> None:
    cfg = load_yaml(config_path)
    fcfg = load_yaml(features_path)

    target = fcfg["target"]

    if feature_set is None:
        feature_set = fcfg["default_set"]

    features = fcfg["feature_sets"][feature_set]

    df = read_dataset(dataset)
    df = clean_dataset(df, target)

    selected, missing = feature_selection(df, features)
    print(
        f"feature_set={feature_set} total={len(features)}",
        f"found={len(selected)} missing={len(missing)}"
    )

    X_train, X_val, X_test, y_train, y_val, y_test, X_train_full, y_train_full = split_data(
        df, selected, target, cfg["split"]
    )

    estimator = load_estimator(cfg)

    best_params = hyperparameter_optimization(estimator, X_train, y_train, cfg)
    model = train_model(estimator, X_train_full, y_train_full, best_params)

    report = {
        "feature_set": feature_set,
        "selected_features": selected,
        "missing_features": missing,
        "best_params": best_params,
        "val": evaluate(model, X_val, y_val),
        "test": evaluate(model, X_test, y_test),
    }

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, outdir / "model.pkl")
    (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    redshift_scatter(y_train_full, model.predict(X_train_full), outdir / "scatter_train.png")
    redshift_scatter(y_val, model.predict(X_val), outdir / "scatter_val.png")
    redshift_scatter(y_test, model.predict(X_test), outdir / "scatter_test.png")
    
    report["plots"] = {
        "train": "scatter_train.png",
        "val": "scatter_val.png",
        "test": "scatter_test.png",
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    print("Saved:", outdir / "scatter_train.png")
    print("Saved:", outdir / "scatter_val.png")
    print("Saved:", outdir / "scatter_test.png")
    print("Saved:", outdir / "model.pkl")
    print("Saved:", outdir / "report.json")


if __name__ == "__main__":
    main()
