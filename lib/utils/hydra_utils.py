from __future__ import annotations

import tempfile
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

CONFIG_GROUPS = ("train", "model", "dataset")


def hydra_to_yaml(cfg: DictConfig) -> Path:
    flat: dict[str, object] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    if isinstance(flat, dict):
        for group in CONFIG_GROUPS:
            sub = flat.pop(group, {})
            if isinstance(sub, dict):
                flat.update(sub)
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        OmegaConf.save(OmegaConf.create(flat), f.name)
        return Path(f.name)
