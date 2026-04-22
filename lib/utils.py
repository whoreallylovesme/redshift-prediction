import yaml
from pydantic import BaseModel


def load_yaml(path: str) -> object:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml_as[ModelT: BaseModel](path: str, model_cls: type[ModelT]) -> ModelT:
    data = load_yaml(path)
    return model_cls.model_validate(data)
