from collections.abc import Callable
from pathlib import Path
from typing import cast

from lib.config import FeaturesConfig, PytorchTrainConfig, SklearnTrainConfig
from lib.dataloaders.dataset import read_dataset
from lib.utils.hydra_utils import run_recipe_cli
from lib.utils.log import get_logger
from lib.utils.registry import TORCH_TRAINERS
from lib.utils.utils import load_yaml, load_yaml_as, pick_device

logger = get_logger()


def _train_sklearn(
    train_path: str,
    config_path: str,
    features_path: str,
    feature_set: str,
    outdir: str,
) -> None:
    from lib.train.sklearn import feature_selection, fit_regressor
    from photoz.metrics import evaluate_regression

    cfg = load_yaml_as(config_path, SklearnTrainConfig)
    fcfg = load_yaml_as(features_path, FeaturesConfig)
    frame = read_dataset(train_path)
    selected, missing = feature_selection(frame, fcfg.feature_sets[feature_set])
    model, report = fit_regressor(
        frame, selected, fcfg.target, cfg, threads=None, metrics=evaluate_regression
    )
    report.update(feature_set=feature_set, missing_features=missing)

    import joblib as jb

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    jb.dump(model, out / "model.pkl")

    import json

    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Saved to {out.resolve()}")


def _train_capsnet(config_path: str, **_: object) -> None:
    from photoz.training.capsnet import run_train

    run_train(Path(config_path))


def _train_lightning_default(
    config_path: str,
    outdir: str | None = None,
    **_: object,
) -> None:
    import torch

    from lib.dataloaders.images import image_loaders
    from lib.train.regression import RegressionTask
    from lib.train.regression_lightning import run_regression
    from lib.utils.estimator_loader import load_estimator
    from lib.workflow.data import symbol

    cfg = PytorchTrainConfig.model_validate(load_yaml(config_path))
    output = Path(outdir or cfg.out_dir or Path("nn_runs") / Path(config_path).stem)
    task_spec = cfg.training_task
    torch.manual_seed(cfg.seed)
    if cfg.image_catalog is not None:
        tr, va = image_loaders(cfg, cfg.image_catalog, output)
    else:
        import numpy as np

        from photoz.data.npz import PhotozDataset, get_npz_dataloaders

        tr, va = get_npz_dataloaders(
            cfg.data_dir,
            cfg.catalog,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            train_ratio=cfg.train_ratio,
            seed=cfg.seed,
            weighted_sampling=cfg.weighted_sampling,
            max_samples=cfg.max_samples,
            crop_size=cfg.crop_size,
            out_size=cfg.out_size,
            physics_augment=cfg.physics_augment,
        )
        output.mkdir(parents=True, exist_ok=True)
        train_images = cast(PhotozDataset, tr.dataset)
        np.savez(output / "image_normalization.npz", mean=train_images.mean, std=train_images.std)
        task_spec = task_spec or "photoz.training.regression:photoz_regression_task"
    estimator = load_estimator(
        cfg.estimator,
        expected_type=torch.nn.Module,
        params=cfg.estimator_params,
    )
    model = cast(torch.nn.Module, estimator)
    regression_task = symbol(task_spec)(cfg) if task_spec else RegressionTask()
    run_regression(
        cfg, model, tr, va, str(output), task=regression_task, device=pick_device(cfg.device)
    )


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
        cast(str, outdir),
    )


def _train_euclid_clip(config_path: str, **_: object) -> None:
    from photoz.training.euclid_clip_finetune import run_train

    run_train(Path(config_path))


def _train_euclid_clip_pretrain(config_path: str, **_: object) -> None:
    from photoz.training.euclid_clip_pretrain import run_train

    run_train(Path(config_path))


def _train_euclid_classical(config_path: str, **_: object) -> None:
    from photoz.training.euclid_classical import run_train

    run_train(Path(config_path))


_ARCHIVED_TRAINERS = {
    "capsnet": _train_capsnet,
    "euclid_clip": _train_euclid_clip,
    "euclid_clip_pretrain": _train_euclid_clip_pretrain,
    "euclid_classical": _train_euclid_classical,
}


def train(
    config_path: str,
    outdir: str | None = None,
    train_path: str | None = None,
    features_path: str | None = None,
    feature_set: str | None = None,
) -> None:
    raw_cfg = load_yaml(config_path)
    recipe = raw_cfg if isinstance(raw_cfg, dict) else {}
    model_type = recipe.get("model_type", "sklearn")
    model = str(recipe.get("model", ""))
    trainers = (
        _ARCHIVED_TRAINERS | TORCH_TRAINERS if recipe.get("image_catalog") is None
        else TORCH_TRAINERS
    )
    if model_type == "lightning":
        trainer: Callable[..., object] = trainers.get(model, _train_lightning_default)
    elif model_type in ("torch", "classical") and model in trainers:
        trainer = trainers[model]
    elif model_type == "sklearn":
        trainer = _train_sklearn_entry
    else:
        raise ValueError(f"unsupported model_type {model_type!r}")
    trainer(
        config_path,
        outdir=outdir,
        train_path=train_path,
        features_path=features_path,
        feature_set=feature_set,
    )


def main() -> None:
    run_recipe_cli(train)


if __name__ == "__main__":
    main()
