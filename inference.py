from typing import Any

import click
import joblib
import pandas as pd
import yaml

from lib.dataset import read_dataset


def load_yaml(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--model", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--test", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--features", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--feature-set", "feature_set", required=True)
@click.option("--out", required=True, type=click.Path(dir_okay=False))
def main(model: str, test: str, features: str, feature_set: str, out: str) -> None:
    fcfg = load_yaml(features)
    target = fcfg["target"]
    feature_list = fcfg["feature_sets"][feature_set]

    df = read_dataset(test)

    X = df[feature_list]
    y = df[target]

    m = joblib.load(model)
    pred = m.predict(X)

    preds = pd.DataFrame({
        "y_true": y.to_numpy(),
        "y_pred": pred,
    })
    preds.to_parquet(out, index=False)


if __name__ == "__main__":
    main()
