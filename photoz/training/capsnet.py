from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from hydra.utils import instantiate

from lib.dataloaders.staged import StagedLoaders
from lib.utils.log import get_logger
from lib.utils.utils import plugin
from photoz.training.capsnet_lightning import CapsNetModule, TrainConfig, run_staged

logger = get_logger()

_TRAIN_CFG_FIELDS = {f for f in TrainConfig.__dataclass_fields__}


def run_train(config_path: Path) -> None:
    with config_path.open() as f:
        raw = yaml.safe_load(f)

    labels = raw.pop("labels", None)
    if isinstance(labels, dict) and labels.get("enabled"):
        generator = labels.get("generator")
        if not isinstance(generator, str):
            raise ValueError("Enabled labels require a generator plugin")
        plugin(generator)(config_path)

    data_config = raw.pop("data", None)
    if not isinstance(data_config, dict) or "_target_" not in data_config:
        raise ValueError("CapsNet data requires a loader factory in data._target_")
    augment_cfg: dict[str, Any] = raw.pop("augment", {})
    raw.pop("inference", None)

    train_kwargs = {k: v for k, v in raw.items() if k in _TRAIN_CFG_FIELDS}
    loader_kwargs = {k: v for k, v in raw.items() if k not in _TRAIN_CFG_FIELDS}
    cfg = TrainConfig(**train_kwargs)
    loaders = instantiate(
        data_config,
        **loader_kwargs,
        seed=cfg.seed,
        crop_size=cfg.crop_size,
        in_channels=cfg.in_channels,
        rot90_prob=float(augment_cfg.get("rot90_prob", 0.67)),
        flip_prob=float(augment_cfg.get("flip_prob", 0.33)),
        include_stage_a=cfg.stage_a_epochs > 0,
    )
    if not isinstance(loaders, StagedLoaders):
        raise TypeError("CapsNet data factory must return StagedLoaders")

    logger.info(f"[train] config: {config_path}  out_dir={cfg.out_dir}")
    torch.manual_seed(cfg.seed)
    module = CapsNetModule(cfg)
    final_ckpt = run_staged(
        module,
        loaders.train_a,
        loaders.val_a,
        loaders.train_b,
        loaders.val_b,
        cfg,
    )
    logger.info(f"[train] done. final checkpoint: {final_ckpt}")
