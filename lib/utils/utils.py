from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import yaml
from pydantic import BaseModel, ImportString, TypeAdapter

if TYPE_CHECKING:
    import torch


# Keep the qualified names in saved recipes. Pydantic resolves either module:attr or module.attr.
symbol = TypeAdapter[Any](ImportString).validate_python
plugin = TypeAdapter[Callable[..., Any]](
    Annotated[Callable[..., Any], ImportString]
).validate_python


def check(ok: object, message: str) -> None:
    if not bool(ok):
        raise ValueError(message)


def load_yaml(path: str | Path) -> object:
    from lib.config.snapshot import record_config

    record_config(path)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml_as[ModelT: BaseModel](path: str | Path, model_cls: type[ModelT]) -> ModelT:
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
