import json
from pathlib import Path
import yaml
import joblib
import click

from lib.dataset import read_dataset, clean_dataset
from lib.estimator_loader import load_estimator
from lib.training import feature_selection, split_data, hyperparameter_optimization, train_model, evaluate

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@click.command()
@click.option("--dataset", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--features", "features_path", required=True, type=click.Path(exists=True))
@click.option("--feature-set", "feature_set", default=None)
@click.option("--outdir", default="runs/last")
def main(dataset, config_path, features_path, feature_set, outdir):
    cfg = load_yaml(config_path)
    fcfg = load_yaml(features_path)

    target = fcfg["target"]

    if feature_set is None:
        feature_set = fcfg["default_set"]

    features = fcfg["feature_sets"][feature_set]

    df = read_dataset(dataset)
    df = clean_dataset(df, target)

    selected, missing = feature_selection(df, features)
    print(f"feature_set={feature_set} total={len(features)} found={len(selected)} missing={len(missing)}")
    if missing:
        print("missing (first 30):", missing[:30])

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

    print("Saved:", outdir / "model.pkl")
    print("Saved:", outdir / "report.json")

if __name__ == "__main__":
    main()
