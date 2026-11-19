from collections.abc import Callable
from pathlib import Path
from typing import Any

from lib.utils.hydra_utils import run_recipe_cli
from lib.utils.registry import TORCH_PREDICTORS
from lib.utils.utils import load_yaml


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


def _select_predictor(recipe: dict[str, Any]) -> Callable[..., object]:
    model_type = recipe.get("model_type", "sklearn")
    model = str(recipe.get("model", ""))
    if model_type == "sklearn":
        from lib.inference.regression import predict_sklearn

        return predict_sklearn
    if model_type in {"lightning", "torch"}:
        predictor = TORCH_PREDICTORS.get(model)
        if predictor is None and model and recipe.get("image_catalog") is None:
            from photoz.methods.legacy import NAMED_PREDICTORS

            predictor = NAMED_PREDICTORS.get(model)
        if predictor is None and model_type == "lightning":
            predictor = TORCH_PREDICTORS.get("", predict_images)
        if predictor is not None:
            return predictor
        raise ValueError(f"unsupported {model_type} model {model!r}")
    raise ValueError(f"unsupported model_type {model_type!r}")


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
    recipe = raw_cfg if isinstance(raw_cfg, dict) else {}
    predictor = _select_predictor(recipe)
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


def main() -> None:
    run_recipe_cli(lambda path: inference(config_path=path))


if __name__ == "__main__":
    main()
