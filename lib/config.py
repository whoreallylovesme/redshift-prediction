from dataclasses import dataclass, field
from typing import Any, Dict, Literal
import yaml

ModelType = Literal["catboost", "local_linear", "mlp", "rf", "gam"]

@dataclass
class ExperimentConfig:
    name: str
    model_type: str
    params: dict = field(default_factory=dict)
    scaler: dict | None = None


def load_experiment_configs(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    items = raw.get("experiments", [])
    configs = []
    for item in items:
        cfg = ExperimentConfig(
            name=item["name"],
            model_type=item["model_type"],
            params=item.get("params", {}),
            scaler=item.get("scaler"),
        )
        configs.append(cfg)
    return configs
    