from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    TypeAdapter,
    ValidationInfo,
    model_validator,
)

from lib.config.schemas import DatasetConfig, FeaturesConfig, PytorchTrainConfig, SklearnTrainConfig
from lib.utils.utils import load_yaml_as, plugin


def preflight[T](inspect: Callable[[T], object]) -> AfterValidator:
    """Inspect recipe references only on --check. Preserve paths and import strings in the model."""

    def validate(value: T, info: ValidationInfo) -> T:
        if info.context and info.context.get("check"):
            inspect(value)
        return value

    return AfterValidator(validate)


def recipe_schema(schema: type[BaseModel]) -> AfterValidator:
    return preflight(partial(load_yaml_as, model_cls=schema))


def identifier(value: str) -> str:
    if not value.replace("-", "_").isidentifier():
        raise ValueError(f"Invalid method or split identifier {value!r}")
    return value


Identifier = Annotated[str, AfterValidator(identifier)]
Plugin = Annotated[str, preflight(plugin)]


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Dataset(Config):
    source: Path
    reader: Plugin | None = None
    target: str | None = None
    features: list[str] = Field(default_factory=list)
    feature_catalog: Path | None = None
    feature_set: str | None = None
    id_column: str | None = None
    csv_id_type: Literal["string", "int64"] = "string"
    weight_column: str | None = None
    split_column: str | None = None
    evaluation_files: dict[Identifier, Path] = Field(default_factory=dict)
    split_config: Annotated[Path, recipe_schema(DatasetConfig)] | None = None
    shuffle: bool = True
    train_label: str = "train"
    test_label: str = "test"
    test_size: float = Field(default=0.2, gt=0, lt=1)
    seed: int = 42
    cleaner: Plugin | None = None

    def target_features(self) -> tuple[str, list[str]]:
        if self.feature_catalog:
            catalog = load_yaml_as(self.feature_catalog, FeaturesConfig)
            if self.feature_set not in catalog.feature_sets:
                raise ValueError(
                    f"Unknown feature set {self.feature_set!r} in {self.feature_catalog}"
                )
            target, features = catalog.target, catalog.feature_sets[str(self.feature_set)]
        else:
            target, features = str(self.target), self.features
        id_column = self.id_column or "__source_row"
        if target == id_column:
            raise ValueError("Dataset identifier and target must be distinct")
        if not features:
            raise ValueError("A regression dataset requires at least one feature")
        reserved = {target, id_column, self.split_column, self.weight_column}
        if len(set(features)) != len(features) or reserved.intersection(features):
            raise ValueError(
                "Features must be unique and exclude target, identifier, weight and split"
            )
        return target, features

    @model_validator(mode="after")
    def columns(self, info: ValidationInfo) -> Self:
        if self.id_column in {"y_true", "y_pred"}:
            raise ValueError("Dataset identifier conflicts with canonical prediction columns")
        if self.feature_catalog is None and (not self.target or not self.features):
            raise ValueError("Specify target and features, or feature_catalog and feature_set")
        if self.feature_catalog is not None and (
            not self.feature_set or self.features or self.target
        ):
            raise ValueError("A feature catalog supplies both target and features")
        random_split = bool({"test_size", "seed", "shuffle"} & self.model_fields_set)
        if self.evaluation_files and (self.split_column or self.split_config or random_split):
            raise ValueError("Explicit evaluation files supply the train/evaluation split")
        if self.split_config and (self.split_column or random_split):
            raise ValueError("A split config supplies test_size, shuffle and seed")
        if self.train_label == self.test_label:
            raise ValueError("Train and test labels must differ")
        if info.context and info.context.get("check"):
            self.target_features()
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


class Tabular(Method):
    kind: Literal["tabular"]
    recipe: Annotated[Path, recipe_schema(SklearnTrainConfig)]


class Saved(Method):
    kind: Literal["saved"]
    predictions: Path


class External(Method):
    kind: Literal["external"]
    predictor: Plugin
    report_from_predictor: bool = False
    resource: Literal["cpu", "accelerator"] = "cpu"
    recipe: Annotated[Path, preflight(TypeAdapter(FilePath).validate_python)]
    options: dict[str, Any] = Field(default_factory=dict)


class ImageSplit(Config):
    images: Path
    catalog: Path
    role: str | None = None


class Lightning(Method):
    kind: Literal["lightning"]
    recipe: Annotated[Path, recipe_schema(PytorchTrainConfig)]
    predict_on: dict[Identifier, ImageSplit] = Field(min_length=1)


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
        if self.dataset is None and any(
            isinstance(method, Tabular) for method in self.methods.values()
        ):
            raise ValueError("A tabular method requires a dataset")
        return self
