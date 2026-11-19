from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, Self, TypedDict

from pydantic import (
    AfterValidator,
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    TypeAdapter,
    ValidationInfo,
    model_validator,
    with_config,
)

from lib.config.schemas import (
    DatasetConfig,
    Estimator,
    FeaturesConfig,
    ImageTraining,
    PytorchTrainConfig,
    SklearnTrainConfig,
)
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


@with_config(extra="allow")
class DatasetFactory(TypedDict):
    """A Hydra callable with its native keyword arguments."""

    _target_: Plugin


class Dataset(Config):
    source: Path | DatasetFactory
    reader: Plugin | None = None
    target: str | None = None
    features: list[str] = Field(default_factory=list)
    feature_catalog: Path | None = None
    feature_set: str | None = None
    id_column: str | None = None
    images: Path | dict[Identifier, Path] | None = None
    image_size: int = Field(default=224, gt=0)
    crop_size: int | None = Field(default=None, gt=0)
    array_key: str = "image"
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


class RecipeMethod(Method):
    # A file reference is supported for previously saved experiments.
    recipe: Path | None = None
    model: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def recipe_path(self) -> Path:
        if self.recipe is None:
            raise ValueError("Inline model parameters have not been resolved for this run")
        return self.recipe

    @model_validator(mode="after")
    def parameter_source(self) -> Self:
        if self.recipe and self.params:
            raise ValueError("Choose inline params or a recipe file")
        if not self.recipe and not self.model:
            raise ValueError("A model implementation or an existing recipe is required")
        return self


class Tabular(RecipeMethod):
    model: Estimator | None = None
    kind: Literal["tabular"]
    recipe: Annotated[Path, recipe_schema(SklearnTrainConfig)] | None = None


class Saved(Method):
    kind: Literal["saved"]
    predictions: Path


class External(RecipeMethod):
    kind: Literal["external"]
    model: Plugin = Field(
        alias="predictor",
        validation_alias=AliasChoices("model", "predictor"),
        serialization_alias="model",
    )
    report_from_predictor: bool = False
    resource: Literal["cpu", "accelerator"] = "cpu"
    recipe: Annotated[Path, preflight(TypeAdapter(FilePath).validate_python)] | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @property
    def predictor(self) -> str:
        return self.model


class ImageSplit(Config):
    images: Path
    catalog: Path
    role: str | None = None


class Lightning(RecipeMethod):
    model: Estimator | None = None
    kind: Literal["lightning"]
    recipe: Annotated[Path, recipe_schema(PytorchTrainConfig)] | None = None
    predict_on: dict[Identifier, ImageSplit] = Field(default_factory=dict)
    train: ImageTraining | None = None
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
                    self.dataset.feature_catalog or (self.dataset.target and self.dataset.features)
                ):
                    raise ValueError("Tabular methods require a dataset with a target and features")
            if isinstance(method, (Lightning, Tabular)) and method.recipe and method.model:
                raise ValueError("Choose model or a historical recipe")
            if isinstance(method, Lightning) and method.recipe:
                if not method.predict_on or (
                    {"train", "device", "use_features"} & method.model_fields_set
                ):
                    raise ValueError("File-based Lightning methods use their recipe and predict_on")
            if isinstance(method, Lightning) and method.recipe is None:
                ds = self.dataset
                if ds is None or not ds.id_column or not (ds.target or ds.feature_catalog):
                    raise ValueError(
                        "Image data requires id_column and a target or feature_catalog"
                    )
                if not method.train or method.predict_on:
                    raise ValueError(
                        "Inline Lightning methods use train and the shared dataset splits"
                    )
                if method.use_features and not (ds.feature_catalog or ds.features):
                    raise ValueError("use_features requires dataset features")
                image_splits(ds, method.split)
        if (
            self.dataset
            and info.context
            and info.context.get("check")
            and any(
                isinstance(m, Tabular) or (isinstance(m, Lightning) and m.recipe is None)
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
