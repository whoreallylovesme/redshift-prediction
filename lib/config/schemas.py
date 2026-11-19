from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    AliasPath,
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationInfo,
    model_validator,
)

from lib.utils.estimator_loader import estimator_name
from lib.utils.utils import load_yaml, load_yaml_as, plugin


def callable_name(value: str) -> str:
    plugin(value)
    return value


Estimator = Annotated[str, AfterValidator(estimator_name)]


class DatasetConfig(BaseModel):
    test_size: float = Field(validation_alias=AliasPath("split", "test_size"))
    shuffle: bool = Field(default=True, validation_alias=AliasPath("split", "shuffle"))
    random_seed: int = Field(default=42, validation_alias=AliasPath("split", "random_seed"))


class FeaturesConfig(BaseModel):
    target: str
    feature_sets: dict[str, list[str]]
    default_set: str | None = None


class SklearnTrainConfig(BaseModel):
    model_type: str | None = None
    estimator: Estimator
    estimator_params: dict[str, Any] = Field(default_factory=dict)
    param_grid: dict[str, list[Any]] = Field(default_factory=dict)
    val_size: float = Field(
        default=0.1, ge=0, lt=1, validation_alias=AliasPath("split", "val_size")
    )
    shuffle: bool = Field(default=True, validation_alias=AliasPath("split", "shuffle"))
    random_seed: int = Field(default=42, validation_alias=AliasPath("split", "random_seed"))
    hyperopt_enabled: bool = Field(default=False, validation_alias=AliasPath("hyperopt", "enabled"))
    hyperopt_cv: int = Field(default=3, validation_alias=AliasPath("hyperopt", "cv"))
    hyperopt_scoring: str = Field(
        default="neg_root_mean_squared_error", validation_alias=AliasPath("hyperopt", "scoring")
    )
    hyperopt_n_jobs: int = Field(default=1, validation_alias=AliasPath("hyperopt", "n_jobs"))
    hyperopt_verbose: int = Field(default=1, validation_alias=AliasPath("hyperopt", "verbose"))
    out_dir: str | None = None
    train_path: str | None = None
    test_path: str | None = None
    features_path: str | None = None
    feature_set: str | None = None


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
    image_catalog: ImageCatalogConfig | None = None
    device: str = "auto"
    training_task: Annotated[str, AfterValidator(callable_name)] | None = None
    model_type: Literal["lightning"]
    estimator: Estimator
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


class EuclidStage(BaseModel):
    config: str
    enabled: bool = True


class EuclidPretrainStage(EuclidStage):
    seeds: list[int] = Field(default_factory=list)


class EuclidFinetuneStage(EuclidStage):
    seed_pairs: list[tuple[int, int]]
    clip_root: str | None = None
    out_dir: str | None = None


class EuclidClassicalStage(EuclidStage):
    learners: list[str] = Field(default_factory=list)


class EuclidFlowConfig(BaseModel):
    pretrain: EuclidPretrainStage | None = None
    finetune: EuclidFinetuneStage
    inference: bool = True
    classical: EuclidClassicalStage | None = None
    report: EuclidStage | None = None


class FlowConfig(BaseModel):
    tracking_uri: str | None = None
    pipeline: Literal["tabular", "nn", "capsnet", "euclid_clip"]
    runs_root: str
    prepared_root: str | None = None
    dataset: dict[str, str] | None = None
    models: list[dict[str, str]] = Field(default_factory=list)
    splits: list[dict[str, str]] = Field(default_factory=list)
    summaries: dict[str, bool | str] = Field(default_factory=dict)
    train: dict[str, str] | None = None
    euclid: EuclidFlowConfig | None = None

    @model_validator(mode="after")
    def referenced_recipes(self, info: ValidationInfo) -> Self:
        if not (info.context and info.context.get("check")):
            return self
        if self.pipeline == "tabular":
            required = {"input_csv", "data_config", "features"}
            if not self.dataset or not required.issubset(self.dataset):
                raise ValueError(
                    "Tabular flows require dataset.input_csv, data_config and features"
                )
            load_yaml_as(self.dataset["data_config"], DatasetConfig)
            catalog = load_yaml_as(self.dataset["features"], FeaturesConfig)
            for model in self.models:
                if model.get("feature_set") not in catalog.feature_sets:
                    raise ValueError(f"Unknown feature set {model.get('feature_set')!r}")
        if self.pipeline in {"tabular", "nn"}:
            schema = SklearnTrainConfig if self.pipeline == "tabular" else PytorchTrainConfig
            for model in self.models:
                if not model.get("config"):
                    raise ValueError("Each legacy model requires a config file")
                load_yaml_as(model["config"], schema)
        if self.pipeline == "capsnet":
            if not self.train or not self.train.get("config"):
                raise ValueError("CapsNet requires train.config")
            TypeAdapter(dict[str, Any]).validate_python(load_yaml(self.train["config"]))
        if self.euclid:
            for stage in (
                self.euclid.pretrain,
                self.euclid.finetune,
                self.euclid.classical,
                self.euclid.report,
            ):
                if stage and stage.enabled:
                    TypeAdapter(dict[str, Any]).validate_python(load_yaml(stage.config))
        return self
