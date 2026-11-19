from pathlib import Path
from typing import cast

import joblib
import pandas as pd

from lib.config import FeaturesConfig, SklearnTrainConfig
from lib.dataloaders.dataset import read_dataset
from lib.utils.utils import load_yaml_as


def _inference_sklearn(
    model_path: str,
    test_path: str,
    features_path: str,
    feature_set: str,
    out: str,
    id_column: str | None = None,
) -> None:
    fcfg = load_yaml_as(features_path, FeaturesConfig)
    target = fcfg.target
    feature_list = fcfg.feature_sets[feature_set]

    df = read_dataset(test_path)
    model = joblib.load(model_path)
    fitted_features = list(getattr(model, "feature_names_in_", feature_list))
    if not set(fitted_features).issubset(feature_list):
        raise ValueError("Prediction feature set differs from the fitted model")
    pred = model.predict(df[fitted_features])

    preds = pd.DataFrame(index=df.index)
    if id_column is not None:
        preds[id_column] = df[id_column]
    if target in df:
        preds["y_true"] = df[target]
    preds["y_pred"] = pred
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)


def predict_sklearn(
    config_path: str,
    model_path: str | None = None,
    out: str | None = None,
    test_path: str | None = None,
    features_path: str | None = None,
    feature_set: str | None = None,
    id_column: str | None = None,
    **_: object,
) -> None:
    scfg = load_yaml_as(config_path, SklearnTrainConfig)
    out_dir = scfg.out_dir or f"runs/{Path(config_path).stem}"
    model_path = model_path or f"{out_dir}/model.pkl"
    out = out or f"{out_dir}/preds.parquet"
    test_path = test_path or scfg.test_path
    features_path = features_path or scfg.features_path
    feature_set = feature_set or scfg.feature_set
    _inference_sklearn(
        cast(str, model_path),
        cast(str, test_path),
        cast(str, features_path),
        cast(str, feature_set),
        out,
        id_column,
    )
