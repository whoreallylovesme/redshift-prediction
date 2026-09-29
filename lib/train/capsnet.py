from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from torch.utils.data import DataLoader

from lib.dataloaders.euclid_capsnet import get_euclid_capsnet_dataloaders
from lib.dataloaders.hdf5 import get_hdf5_dataloaders
from lib.dataloaders.teddy_capsnet import get_teddy_capsnet_dataloaders
from lib.train.capsnet_lightning import CapsNetModule, TrainConfig, run_staged
from lib.utils.log import get_logger

logger = get_logger()

_TRAIN_CFG_FIELDS = {f for f in TrainConfig.__dataclass_fields__}


def run_train(config_path: Path) -> None:
    with config_path.open() as f:
        raw = yaml.safe_load(f)

    if isinstance(raw.get("labels"), dict) and raw["labels"].get("enabled"):
        from lib.train.labels import run_generate

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
    elif dataset == "euclid":
        train_a, val_a, train_b, val_b = _build_euclid(
            loader_kwargs, cfg, rot90_prob, flip_prob,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    module = CapsNetModule(cfg)
    init_from = raw.get("init_from") if isinstance(raw, dict) else None
    if init_from:
        _init_from_pretrained(module, Path(init_from))
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


def _build_euclid(
    lk: dict[str, Any], cfg: TrainConfig,
    rot90_prob: float, flip_prob: float,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    train_a, val_a, train_b, val_b = get_euclid_capsnet_dataloaders(
        stamps_dir=lk["stamps_dir"],
        labels_parquet=lk["labels_parquet"],
        train_tiles=list(lk["train_tiles"]),
        valid_tiles=list(lk["valid_tiles"]),
        stage_a_zprob=float(lk.get("stage_a_zprob", 0.95)),
        save_scales_to=str(Path(cfg.out_dir) / "weights" / "norm_scales.npy"),
        batch_size=lk.get("batch_size", 256),
        num_workers=lk.get("num_workers", 3),
        z_min=cfg.z_min,
        z_max=cfg.z_max,
        seed=cfg.seed,
        rot90_prob=rot90_prob,
        flip_prob=flip_prob,
    )
    return train_a, val_a, train_b, val_b


def _init_from_pretrained(module: CapsNetModule, ckpt_path: Path) -> None:
    """Перенос SDSS-капснета (6ch) на Euclid (4ch) по длинам волн.

    VIS<-mean(r,i), Y<-z, J<-mean(z,W1), H<-W1; остальные слои как есть.
    """
    import torch

    sd = torch.load(ckpt_path, map_location="cpu")
    state = module.net.state_dict()
    loaded, skipped = 0, []
    conv1 = sd.get("conv1.weight")
    if conv1 is not None and conv1.shape[1] == 6 and state["conv1.weight"].shape[1] == 4:
        # порядок каналов претрена: g, r, i, z, W1, W2
        mapping = {
            0: (conv1[:, 1] + conv1[:, 2]) / 2,  # VIS <- (r+i)/2
            1: conv1[:, 3].clone(),               # Y <- z
            2: (conv1[:, 3] + conv1[:, 4]) / 2,   # J <- (z+W1)/2
            3: conv1[:, 4].clone(),               # H <- W1
        }
        new_conv = torch.stack([mapping[i] for i in range(4)], dim=1)
        state["conv1.weight"] = new_conv
        loaded += 1
    for k, v in sd.items():
        if k in ("conv1.weight",):
            continue
        if k in state and state[k].shape == v.shape:
            state[k] = v
            loaded += 1
        else:
            skipped.append(k)
    module.net.load_state_dict(state)
    logger.info(
        f"[train] претрен {ckpt_path.name}: перенесено {loaded} тензоров, "
        f"пропущено {len(skipped)}{', первый: ' + skipped[0] if skipped else ''}",
    )
