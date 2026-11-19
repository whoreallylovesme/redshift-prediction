from functools import partial
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from lib.models.registry import NN_FACTORIES, TABULAR_FACTORIES
from lib.utils.estimator_loader import estimator_name
from lib.utils.utils import plugin


def callable_name(value: str) -> str:
    plugin(value)
    return value


TabularEstimator = Annotated[
    str, AfterValidator(partial(estimator_name, factories=TABULAR_FACTORIES))
]
NeuralEstimator = Annotated[str, AfterValidator(partial(estimator_name, factories=NN_FACTORIES))]


class TabularSplit(BaseModel, extra="forbid"):
    val_size: float = Field(default=0.1, ge=0, lt=1)
    shuffle: bool = True
    random_seed: int = 42


class TabularSearch(BaseModel, extra="forbid"):
    enabled: bool = False
    cv: int = Field(default=3, ge=2)
    scoring: str = "neg_root_mean_squared_error"
    n_jobs: int = 1
    verbose: int = Field(default=1, ge=0)


class TabularTraining(BaseModel, extra="forbid"):
    split: TabularSplit = Field(default_factory=TabularSplit)
    hyperopt: TabularSearch = Field(default_factory=TabularSearch)
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)


class SklearnTrainConfig(TabularTraining):
    model_type: str | None = None
    estimator: TabularEstimator
    estimator_params: dict[str, Any] = Field(default_factory=dict)


class ImageCatalogConfig(BaseModel):
    features: list[str] = Field(default_factory=list)
    id_column: str
    target_column: str
    array_key: str = "image"
    id_dtype: Literal["string", "int64"] | None = None

    @model_validator(mode="after")
    def distinct_columns(self) -> Self:
        names = [self.id_column, self.target_column, *self.features]
        if len(names) != len(set(names)):
            raise ValueError("Image ID, regression target, and feature columns must be distinct")
        return self


class ImageTraining(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    lr: float = Field(ge=0)
    weight_decay: float = 0.0
    freeze_epochs: int = 0
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    seed: int = 42
    num_workers: int = Field(default=4, ge=0)
    weighted_sampling: bool = False
    z_weight_alpha: float = 0.0
    label_noise: float = 0.0
    head_prefix: str = "classifier"
    max_samples: int | None = None


class PytorchTrainConfig(ImageTraining):
    model_config = ConfigDict(extra="ignore")

    estimator_params: dict[str, Any] = Field(default_factory=dict)
    image_catalog: ImageCatalogConfig
    device: str = "auto"
    training_task: Annotated[str, AfterValidator(callable_name)] | None = None
    model_type: Literal["lightning"]
    estimator: NeuralEstimator
    data_dir: str
    catalog: str
    out_dir: str | None = None
    norm_from: str | None = None
    crop_size: int | None = None
    out_size: int = 224
    physics_augment: bool = False

    @model_validator(mode="before")
    @classmethod
    def training_fields(cls, raw: Any) -> Any:
        if isinstance(raw, dict) and isinstance(raw.get("train"), dict):
            return {
                **raw,
                **{k: v for k, v in raw["train"].items() if k in ImageTraining.model_fields},
            }
        return raw
