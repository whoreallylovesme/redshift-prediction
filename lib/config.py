from dataclasses import dataclass, field
from typing import Any, Dict, Literal
import yaml

ModelType = Literal["catboost", "local_linear", "mlp"]

@dataclass
class ExperimentConfig:
    name: str
    model_type: ModelType
    params: Dict[str, Any] = field(default_factory=dict)

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
        )
        configs.append(cfg)
    return configs
    