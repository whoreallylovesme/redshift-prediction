from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

CONFIG_GROUPS = ("train", "model", "dataset")


def hydra_to_yaml(cfg: DictConfig) -> Path:
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise ValueError("Expected Hydra configuration groups")
    groups: dict[str, Any] = {str(key): value for key, value in values.items()}
    train, model, dataset = (groups.pop(k, {}) for k in CONFIG_GROUPS)
    if "recipe" in model:
        if dataset.get("dataset") != "teddy":
            raise ValueError(f"{model['recipe']} requires the Teddy image catalogue")
        base = OmegaConf.load(model["recipe"])
        overrides = {k: v for k, v in train.items() if k in base.train}
        overrides["seed"] = groups.pop("seed", base.train.seed)
        result = OmegaConf.merge(base, {"train": overrides}, groups)
    else:
        result = OmegaConf.create({**groups, **train, **model, **dataset})
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as stream:
        OmegaConf.save(result, stream)
        return Path(stream.name)
