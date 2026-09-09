from __future__ import annotations

import torch
import yaml
from pydantic import BaseModel


def load_yaml(path: str) -> object:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml_as[ModelT: BaseModel](path: str, model_cls: type[ModelT]) -> ModelT:
    data = load_yaml(path)
    return model_cls.model_validate(data)


def pick_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
