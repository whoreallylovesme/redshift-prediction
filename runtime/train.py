import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import click
import joblib
from sklearn.model_selection import train_test_split

from lib.config_models import FeaturesConfig, PytorchTrainConfig, SklearnTrainConfig
from lib.dataset import read_dataset
from lib.estimator_loader import load_estimator
from lib.log import get_logger
from lib.registry import TORCH_TRAINERS, register_torch_trainer
from lib.training import (
    evaluate,
    feature_selection,
    hyperparameter_optimization,
    train_model,
)
from lib.utils import load_yaml, load_yaml_as

logger = get_logger()



def _train_sklearn(
    train_path: str, config_path: str, features_path: str, feature_set: str, outdir: str,
) -> None:
    cfg = load_yaml_as(config_path, SklearnTrainConfig)
    fcfg = load_yaml_as(features_path, FeaturesConfig)
    target = fcfg.target
    feature_list = fcfg.feature_sets[feature_set]

    df = read_dataset(train_path)
    selected, missing = feature_selection(df, feature_list)

    X = df[selected]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=cfg.val_size,
        random_state=cfg.random_seed,
        shuffle=cfg.shuffle,
    )

    estimator = load_estimator(cfg.estimator)
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
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Saved to {out.resolve()}")

def _train_capsnet(config_path: str, **_: object) -> None:
    from lib.nn.train.capsnet import run_train

    run_train(Path(config_path))

def _train_lightning_default(
    config_path: str, outdir: str | None = None, **_: object,
) -> None:
    import torch.nn as nn

    from lib.estimator_loader import load_estimator
    from lib.nn.dataloaders.npz import get_npz_dataloaders
    from lib.nn.train.regression_lightning import run_regression

    cfg = PytorchTrainConfig.model_validate(load_yaml(config_path))
    outdir = outdir or cfg.out_dir or f"nn_runs/{Path(config_path).stem}"
    loaded = load_estimator(cfg.estimator, expected_type=nn.Module)
    model = cast(nn.Module, loaded)
    tr, va = get_npz_dataloaders(
        cfg.data_dir, cfg.catalog,
        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
        train_ratio=cfg.train_ratio, seed=cfg.seed,
        weighted_sampling=cfg.weighted_sampling, max_samples=cfg.max_samples,
        crop_size=cfg.crop_size, out_size=cfg.out_size,
        physics_augment=cfg.physics_augment,
    )
    run_regression(cfg, model, tr, va, outdir)

def _train_sklearn_entry(
    config_path: str,
    outdir: str | None = None,
    train_path: str | None = None,
    features_path: str | None = None,
    feature_set: str | None = None,
    **_: object,
) -> None:
    scfg = load_yaml_as(config_path, SklearnTrainConfig)
    train_path = train_path or scfg.train_path
    features_path = features_path or scfg.features_path
    feature_set = feature_set or scfg.feature_set
    outdir = outdir or scfg.out_dir or f"runs/{Path(config_path).stem}"
    _train_sklearn(
        cast(str, train_path),
        config_path,
        cast(str, features_path),
        cast(str, feature_set),
        outdir,
    )

register_torch_trainer("capsnet", _train_capsnet)


def train(
    config_path: str,
    outdir: str | None = None,
    train_path: str | None = None,
    features_path: str | None = None,
    feature_set: str | None = None,
) -> None:
    raw_cfg = load_yaml(config_path)
    model_type = raw_cfg.get("model_type", "sklearn") if isinstance(raw_cfg, dict) else "sklearn"
    if model_type == "lightning":
        model = str(raw_cfg.get("model", "")) if isinstance(raw_cfg, dict) else ""
        trainer: Callable[..., object] = TORCH_TRAINERS.get(model, _train_lightning_default)
    elif model_type == "sklearn":
        trainer = _train_sklearn_entry
    else:
        raise ValueError(f"unsupported model_type {model_type!r}")
    trainer(
        config_path, outdir=outdir, train_path=train_path,
        features_path=features_path, feature_set=feature_set,
    )

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, dir_okay=False))
def main(config: str) -> None:
    train(config)

if __name__ == "__main__":
    main()
