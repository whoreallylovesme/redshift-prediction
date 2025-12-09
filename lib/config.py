from dataclasses import dataclass, field
from typing import Any, Dict, Literal

ModelType = Literal["catboost", "local_linear", "mlp"]

@dataclass
class ExperimentConfig:
    name: str
    model_type: ModelType
    params: Dict[str, Any] = field(default_factory=dict)
