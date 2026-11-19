from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd

from lib.config import FeaturesConfig, SklearnTrainConfig
from lib.dataloaders.dataset import read_dataset
from lib.utils.registry import TORCH_PREDICTORS
from lib.utils.utils import load_yaml, load_yaml_as


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


def _infer_sklearn_entry(
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


def predict_images(
    config_path: str,
    model_path: str | None = None,
    out: str | None = None,
    data_dir: str | None = None,
    catalog: str | None = None,
    id_column: str | None = None,
    held_out: bool = False,
    **_: Any,
) -> None:
    from lib.config import PytorchTrainConfig

    cfg = PytorchTrainConfig.model_validate(load_yaml(config_path))
    folder = Path(cfg.out_dir or f"nn_runs/{Path(config_path).stem}")
    weights = Path(model_path) if model_path else folder / "best_model.pt"
    target = Path(out) if out else folder / "preds.parquet"
    images, labels = Path(data_dir or cfg.data_dir), Path(catalog or cfg.catalog)
    if cfg.image_catalog:
        from lib.inference.images import predict_catalog

        predict_catalog(
            cfg, cfg.image_catalog, weights, images, labels, target, id_column, held_out,
        )
        return
    from photoz.methods.teddy_images import predict_teddy

    predict_teddy(str(weights), cfg, str(images), str(labels), str(target), id_column)


def inference(
    model_path: str | None = None,
    out: str | None = None,
    config_path: str | None = None,
    test_path: str | None = None,
    features_path: str | None = None,
    feature_set: str | None = None,
    data_dir: str | None = None,
    catalog: str | None = None,
    id_column: str | None = None,
    held_out: bool = False,
) -> None:
    if id_column in {"y_true", "y_pred"}:
        raise ValueError("ID column must be distinct from y_true and y_pred")
    if config_path is None:
        raise ValueError("config_path is required")
    raw_cfg = load_yaml(config_path)
    model_type = raw_cfg.get("model_type", "sklearn") if isinstance(raw_cfg, dict) else "sklearn"
    model = str(raw_cfg.get("model", "")) if isinstance(raw_cfg, dict) else ""
    if model and model_type in {"lightning", "torch"} and model not in TORCH_PREDICTORS:
        from photoz.methods.legacy import register_predictors

        register_predictors()
    if model_type == "lightning":
        predictor: Callable[..., object] = TORCH_PREDICTORS.get(
            model,
            TORCH_PREDICTORS.get("", predict_images),
        )
    elif model_type == "torch" and model and model in TORCH_PREDICTORS:
        predictor = TORCH_PREDICTORS[model]
    elif model_type == "sklearn":
        predictor = _infer_sklearn_entry
    else:
        raise ValueError(f"unsupported model_type {model_type!r}")
    predictor(
        config_path,
        model_path=model_path,
        out=out,
        test_path=test_path,
        features_path=features_path,
        feature_set=feature_set,
        data_dir=data_dir,
        catalog=catalog,
        **({"id_column": id_column} if id_column is not None else {}),
        **({"held_out": True} if held_out else {}),
    )
