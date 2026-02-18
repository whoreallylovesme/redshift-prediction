import json
from pathlib import Path
from typing import Any

import click
import joblib
import yaml
from sklearn.model_selection import train_test_split

from lib.dataset import read_dataset
from lib.estimator_loader import load_estimator
from lib.training import (
    evaluate,
    feature_selection,
    hyperparameter_optimization,
    train_model,
)


def load_yaml(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--train", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--features", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--feature-set", required=True)
@click.option("--outdir", required=True, type=click.Path(file_okay=False))
def main(
    train: str,
    config: str,
    features: str,
    feature_set: str,
    outdir: str
) -> None:
    cfg = load_yaml(config)
    fcfg = load_yaml(features)

    target = fcfg["target"]
    feature_list = fcfg["feature_sets"][feature_set]

    split = cfg["split"]
    val_size = float(split["val_size"])
    shuffle = bool(split["shuffle"])
    random_seed = int(split["random_seed"])

    df = read_dataset(train)

    selected, missing = feature_selection(df, feature_list)

    X = df[selected]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_seed,
        shuffle=shuffle,
    )

    estimator = load_estimator(cfg)

    best_params = hyperparameter_optimization(estimator, X_train, y_train, cfg)
    model = train_model(estimator, X_train, y_train, best_params)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out / "model.pkl")

    report = {
        "feature_set": feature_set,
        "selected_features": selected,
        "missing_features": missing,
        "best_params": best_params,
        "val_metrics": evaluate(model, X_val, y_val),
    }

    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
