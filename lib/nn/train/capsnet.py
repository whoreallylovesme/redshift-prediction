from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from lib.log import get_logger
from lib.nn.dataloaders.hdf5 import get_hdf5_dataloaders
from lib.nn.dataloaders.teddy_capsnet import get_teddy_capsnet_dataloaders
from lib.nn.train.capsnet_lightning import CapsNetModule, TrainConfig, run_staged

logger = get_logger()

_TRAIN_CFG_FIELDS = {f for f in TrainConfig.__dataclass_fields__}


def run_train(config_path: Path) -> None:
    with config_path.open() as f:
        raw = yaml.safe_load(f)

    if isinstance(raw.get("labels"), dict) and raw["labels"].get("enabled"):
        from lib.nn.train.labels import run_generate

        run_generate(config_path)

    dataset = raw.pop("dataset", "hayat")
    raw.pop("model_type", None)
    augment_cfg: dict[str, Any] = raw.pop("augment", {})
    inference_cfg: dict[str, Any] = raw.pop("inference", {})
    max_samples = inference_cfg.get("max_samples")

    train_kwargs = {k: v for k, v in raw.items() if k in _TRAIN_CFG_FIELDS}
    loader_kwargs = {k: v for k, v in raw.items() if k not in _TRAIN_CFG_FIELDS}
    cfg = TrainConfig(**train_kwargs)
    rot90_prob = float(augment_cfg.get("rot90_prob", 0.67))
    flip_prob = float(augment_cfg.get("flip_prob", 0.33))

    logger.info(f"[train] config: {config_path}  dataset={dataset}  out_dir={cfg.out_dir}")
    if max_samples is not None:
        logger.info(f"[train] max_samples: {max_samples}")

    if dataset == "hayat":
        train_a, val_a, train_b, val_b = _build_hayat(
            loader_kwargs, cfg, max_samples, rot90_prob, flip_prob,
        )
    elif dataset == "teddy":
        train_a, val_a, train_b, val_b = _build_teddy(
            loader_kwargs, cfg, max_samples, rot90_prob, flip_prob,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    module = CapsNetModule(cfg)
    final_ckpt = run_staged(module, train_a, val_a, train_b, val_b, cfg)
    logger.info(f"[train] done. final checkpoint: {final_ckpt}")

def _build_hayat(
    lk: dict[str, Any], cfg: TrainConfig, max_samples: int | None,
    rot90_prob: float, flip_prob: float,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    common = dict(
        h5_path=lk["h5_path"],
        morph_parquet=lk["morph_parquet"],
        batch_size=lk.get("batch_size", 400),
        num_workers=lk.get("num_workers", 4),
        train_ratio=lk.get("train_ratio", 0.8),
        seed=cfg.seed,
        max_samples=max_samples,
        weighted_sampling=lk.get("weighted_sampling", False),
        crop_size=cfg.crop_size,
        image_scale=lk.get("image_scale", 10.0),
        rot90_prob=rot90_prob,
        flip_prob=flip_prob,
    )
    logger.info("[cli] building Hayat stage-A loaders (labeled GZ1 subset)")
    train_a, val_a = get_hdf5_dataloaders(**common, labeled_only=True)
    logger.info("[cli] building Hayat stage-B loaders (full set)")
    train_b, val_b = get_hdf5_dataloaders(**common, labeled_only=False)
    return train_a, val_a, train_b, val_b

def _build_teddy(
    lk: dict[str, Any], cfg: TrainConfig, max_samples: int | None,
    rot90_prob: float, flip_prob: float,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    common = dict(
        images_dir=lk["images_dir"],
        catalog_path=lk["catalog_path"],
        gz1_parquet=lk["gz1_parquet"],
        batch_size=lk.get("batch_size", 256),
        num_workers=lk.get("num_workers", 4),
        train_ratio=lk.get("train_ratio", 0.85),
        seed=cfg.seed,
        max_samples=max_samples,
        crop_size=cfg.crop_size,
        in_channels=cfg.in_channels,
        image_scale=lk.get("image_scale", 10.0),
        per_channel_norm=lk.get("per_channel_norm", False),
        rot90_prob=rot90_prob,
        flip_prob=flip_prob,
    )
    logger.info("[cli] building Teddy stage-A loaders (labeled GZ1 subset, class-balanced)")
    train_a, val_a = get_teddy_capsnet_dataloaders(
        **common, weighted_sampling=True, labeled_only=True,
    )
    logger.info("[cli] building Teddy stage-B loaders (full set, z-balanced)")
    train_b, val_b = get_teddy_capsnet_dataloaders(
        **common, weighted_sampling=lk.get("weighted_sampling", True), labeled_only=False,
    )
    return train_a, val_a, train_b, val_b
