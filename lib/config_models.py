from typing import Any, Literal

from pydantic import AliasPath, BaseModel, Field


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
    val_size: float = Field(validation_alias=AliasPath("split", "val_size"))
    shuffle: bool = Field(default=True, validation_alias=AliasPath("split", "shuffle"))
    random_seed: int = Field(default=42, validation_alias=AliasPath("split", "random_seed"))
    hyperopt_enabled: bool = Field(validation_alias=AliasPath("hyperopt", "enabled"))
    hyperopt_cv: int = Field(validation_alias=AliasPath("hyperopt", "cv"))
    hyperopt_scoring: str = Field(validation_alias=AliasPath("hyperopt", "scoring"))
    hyperopt_n_jobs: int = Field(validation_alias=AliasPath("hyperopt", "n_jobs"))
    hyperopt_verbose: int = Field(validation_alias=AliasPath("hyperopt", "verbose"))
    out_dir: str | None = None
    train_path: str | None = None
    test_path: str | None = None
    features_path: str | None = None
    feature_set: str | None = None

class PytorchTrainConfig(BaseModel):
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

class FlowConfig(BaseModel):
    pipeline: Literal["tabular", "nn", "capsnet"]
    runs_root: str
    prepared_root: str | None = None
    dataset: dict[str, str] | None = None
    models: list[dict[str, str]] = Field(default_factory=list)
    splits: list[dict[str, str]] = Field(default_factory=list)
    summaries: dict[str, bool | str] = Field(default_factory=dict)
    train: dict[str, str] | None = None
