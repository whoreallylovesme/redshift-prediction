from typing import Any, Literal, Self

from pydantic import AliasPath, BaseModel, Field, model_validator


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
    estimator: str
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


class PytorchTrainConfig(BaseModel):
    estimator_params: dict[str, Any] = Field(default_factory=dict)
    image_catalog: ImageCatalogConfig | None = None
    device: str = "auto"
    training_task: str | None = None
    model_type: Literal["lightning"]
    estimator: str
    data_dir: str
    catalog: str
    out_dir: str | None = None
    norm_from: str | None = None
    crop_size: int | None = None
    out_size: int = 224
    physics_augment: bool = False
    epochs: int = Field(validation_alias=AliasPath("train", "epochs"))
    batch_size: int = Field(validation_alias=AliasPath("train", "batch_size"))
    lr: float = Field(validation_alias=AliasPath("train", "lr"))
    weight_decay: float = Field(validation_alias=AliasPath("train", "weight_decay"))
    freeze_epochs: int = Field(validation_alias=AliasPath("train", "freeze_epochs"))
    train_ratio: float = Field(default=0.8, validation_alias=AliasPath("train", "train_ratio"))
    seed: int = Field(default=42, validation_alias=AliasPath("train", "seed"))
    num_workers: int = Field(default=4, validation_alias=AliasPath("train", "num_workers"))
    weighted_sampling: bool = Field(
        default=False,
        validation_alias=AliasPath("train", "weighted_sampling"),
    )
    z_weight_alpha: float = Field(
        default=0.0,
        validation_alias=AliasPath("train", "z_weight_alpha"),
    )
    label_noise: float = Field(default=0.0, validation_alias=AliasPath("train", "label_noise"))
    head_prefix: str = Field(
        default="classifier",
        validation_alias=AliasPath("train", "head_prefix"),
    )
    max_samples: int | None = Field(
        default=None,
        validation_alias=AliasPath("train", "max_samples"),
    )


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
