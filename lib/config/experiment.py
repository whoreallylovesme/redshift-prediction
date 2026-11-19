from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypedDict

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    model_validator,
    with_config,
)

from lib.config.schemas import ImageTraining, NeuralEstimator, TabularEstimator, TabularTraining
from lib.utils.utils import plugin


def preflight[T](inspect: Callable[[T], object]) -> AfterValidator:
    """Inspect recipe references only on --check. Preserve paths and import strings in the model."""

    def validate(value: T, info: ValidationInfo) -> T:
        if info.context and info.context.get("check"):
            inspect(value)
        return value

    return AfterValidator(validate)


def identifier(value: str) -> str:
    if not value.replace("-", "_").isidentifier():
        raise ValueError(f"Invalid method or split identifier {value!r}")
    return value


Identifier = Annotated[str, AfterValidator(identifier)]
Plugin = Annotated[str, preflight(plugin)]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")


@with_config(extra="allow")
class DatasetFactory(TypedDict):
    """A Hydra callable with its native keyword arguments."""

    _target_: Plugin


class Dataset(Config):
    source: Path | DatasetFactory
    reader: Plugin | None = None
    target: str | None = None
    features: list[str] = Field(default_factory=list)
    id_column: str | None = None
    images: Path | dict[Identifier, Path] | None = None
    image_size: int = Field(default=224, gt=0)
    crop_size: int | None = Field(default=None, gt=0)
    array_key: str = "image"
    csv_id_type: Literal["string", "int64"] = "string"
    weight_column: str | None = None
    split_column: str | None = None
    evaluation_files: dict[Identifier, Path] = Field(default_factory=dict)
    shuffle: bool = True
    train_label: str = "train"
    test_label: str = "test"
    test_size: float = Field(default=0.2, gt=0, lt=1)
    seed: int = 42
    cleaner: Plugin | None = None

    def target_features(self) -> tuple[str, list[str]]:
        if not self.target:
            raise ValueError("A regression dataset requires a target")
        target, features = self.target, self.features
        id_column = self.id_column or "__source_row"
        if target == id_column:
            raise ValueError("Dataset identifier and target must be distinct")
        if not features and self.images is None:
            raise ValueError("A regression dataset requires at least one feature")
        reserved = {target, id_column, self.split_column, self.weight_column}
        if len(set(features)) != len(features) or reserved.intersection(features):
            raise ValueError(
                "Features must be unique and exclude target, identifier, weight and split"
            )
        return target, features

    @model_validator(mode="after")
    def columns(self) -> Self:
        if not isinstance(self.source, Path) and self.reader and not self.evaluation_files:
            raise ValueError("reader applies to files; the dataset factory reads its own inputs")
        if self.id_column in {"y_true", "y_pred"}:
            raise ValueError("Dataset identifier conflicts with canonical prediction columns")
        random_split = bool({"test_size", "seed", "shuffle"} & self.model_fields_set)
        if self.evaluation_files and (self.split_column or random_split):
            raise ValueError("Explicit evaluation files supply the train/evaluation split")
        if self.split_column and random_split:
            raise ValueError("Choose split_column or random split parameters")
        if self.train_label == self.test_label:
            raise ValueError("Train and test labels must differ")
        return self


class PredictionColumns(Config):
    id_column: str | None = None
    prediction_column: str = "y_pred"
    truth_column: str | None = "y_true"
    reference: Path | None = None
    reference_id_column: str | None = None
    reference_truth_column: str | None = None
    thumbnail_column: str | None = None
    posterior_column: str | None = None
    posterior_grid: list[float] | None = None
    nominal_coverage: float | None = Field(default=None, gt=0, lt=1)
    reference_kind: Literal["truth", "agreement"] | None = None
    history: Path | None = None
    group_column: str | None = None
    slice_bins: dict[str, list[float]] = Field(default_factory=dict)
    modality_columns: dict[str, str] = Field(default_factory=dict)
    example_columns: list[str] = Field(default_factory=list)
    feature_columns: list[str] = Field(default_factory=list)
    slices: list[str] = Field(default_factory=list)
    lower_column: str | None = None
    upper_column: str | None = None
    admitted_column: str | None = None
    status_column: str | None = None
    ok_statuses: list[str | int] | None = None
    role: str | None = None


class Method(Config):
    split: str = "test"
    report: PredictionColumns = Field(default_factory=PredictionColumns)


class ModelMethod(Method):
    params: dict[str, Any] = Field(default_factory=dict)


class Tabular(ModelMethod):
    kind: Literal["tabular"]
    model: TabularEstimator
    train: TabularTraining = Field(default_factory=TabularTraining)


class Saved(Method):
    kind: Literal["saved"]
    predictions: Path


class External(ModelMethod):
    kind: Literal["external"]
    model: Plugin
    report_from_predictor: bool = False
    resource: Literal["cpu", "accelerator"] = "cpu"
    options: dict[str, Any] = Field(default_factory=dict)


class ImageSplit(Config):
    images: Path
    catalog: Path


class Lightning(ModelMethod):
    model: NeuralEstimator
    kind: Literal["lightning"]
    train: ImageTraining
    device: str = "auto"
    use_features: bool = False


MethodConfig = Annotated[Tabular | Saved | External | Lightning, Field(discriminator="kind")]


class Experiment(Config):
    name: str
    output: Path
    task: str = "regression"
    dataset: Dataset | None = None
    methods: dict[Identifier, MethodConfig] = Field(min_length=1)
    bootstrap_replicates: int = Field(default=0, ge=0)
    workers: int = Field(default=1, ge=1)
    threads_per_method: int = Field(default=1, ge=1)
    cache: bool = True
    tracking_uri: str | None = None

    @model_validator(mode="after")
    def methods_and_data(self, info: ValidationInfo) -> Self:
        if info.context and info.context.get("check"):
            from lib.report.task import metric_suite

            metric_suite(self.task)
        for method in self.methods.values():
            if isinstance(method, Tabular):
                if self.dataset is None or not (
                    self.dataset.target and self.dataset.features
                ):
                    raise ValueError("Tabular methods require a dataset with a target and features")
            if isinstance(method, Lightning):
                ds = self.dataset
                if ds is None or not ds.id_column or not ds.target:
                    raise ValueError(
                        "Image data requires id_column and target"
                    )
                if method.use_features and not ds.features:
                    raise ValueError("use_features requires dataset features")
                image_splits(ds, method.split)
        if (
            self.dataset
            and info.context
            and info.context.get("check")
            and any(
                isinstance(m, (Tabular, Lightning))
                for m in self.methods.values()
            )
        ):
            self.dataset.target_features()
        return self


def image_splits(dataset: Dataset, split: str) -> dict[str, Path]:
    labels = {"train", *(dataset.evaluation_files or [split])}
    if "train" in dataset.evaluation_files:
        raise ValueError("The train image split name is reserved")
    if isinstance(dataset.images, Path):
        return dict.fromkeys(labels, dataset.images)
    if dataset.images is None or dataset.images.keys() != labels:
        raise ValueError(f"Image directories must cover the configured splits: {sorted(labels)}")
    return dataset.images
