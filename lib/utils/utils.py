from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    import torch


def check(ok: object, message: str) -> None:
    if not bool(ok):
        raise ValueError(message)


def load_yaml(path: str) -> object:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml_as[ModelT: BaseModel](path: str, model_cls: type[ModelT]) -> ModelT:
    data = load_yaml(path)
    return model_cls.model_validate(data)


def pick_device(spec: str = "auto") -> torch.device:
    import torch

    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
