from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Dataset(Config):
    source: Path
    reader: str | None = None
    target: str | None = None
    features: list[str] = Field(default_factory=list)
    feature_catalog: Path | None = None
    feature_set: str | None = None
    id_column: str | None = None
    csv_id_type: Literal["string", "int64"] = "string"
    weight_column: str | None = None
    split_column: str | None = None
    evaluation_files: dict[str, Path] = Field(default_factory=dict)
    split_config: Path | None = None
    shuffle: bool = True
    train_label: str = "train"
    test_label: str = "test"
    test_size: float = Field(default=0.2, gt=0, lt=1)
    seed: int = 42
    cleaner: str | None = None

    @model_validator(mode="after")
    def columns(self) -> Self:
        if self.id_column in {"y_true", "y_pred"}:
            raise ValueError("Dataset identifier conflicts with canonical prediction columns")
        if self.feature_catalog is None and (not self.target or not self.features):
            raise ValueError("Specify target and features, or feature_catalog and feature_set")
        if self.feature_catalog is not None and (
            not self.feature_set or self.features or self.target
        ):
            raise ValueError("A feature catalog supplies both target and features")
        if self.evaluation_files:
            if (
                self.split_column
                or self.split_config
                or {"test_size", "seed", "shuffle"} & self.model_fields_set
            ):
                raise ValueError("Explicit evaluation files supply the train/evaluation split")
            if any(not k.replace("-", "_").isidentifier() for k in self.evaluation_files):
                raise ValueError("Evaluation split names must be identifiers")
        if self.split_config and (
            self.split_column or {"test_size", "seed", "shuffle"} & self.model_fields_set
        ):
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


class Tabular(Method):
    kind: Literal["tabular"]
    recipe: Path


class Saved(Method):
    kind: Literal["saved"]
    predictions: Path


class External(Method):
    kind: Literal["external"]
    predictor: str
    report_from_predictor: bool = False
    resource: Literal["cpu", "accelerator"] = "cpu"
    recipe: Path
    options: dict[str, Any] = Field(default_factory=dict)


class ImageSplit(Config):
    images: Path
    catalog: Path
    role: str | None = None


class Lightning(Method):
    kind: Literal["lightning"]
    recipe: Path
    predict_on: dict[str, ImageSplit]


MethodConfig = Annotated[Tabular | Saved | External | Lightning, Field(discriminator="kind")]


class Experiment(Config):
    name: str
    output: Path
    task: str = "regression"
    dataset: Dataset | None = None
    methods: dict[str, MethodConfig]
    bootstrap_replicates: int = Field(default=0, ge=0)
    workers: int = Field(default=1, ge=1)
    threads_per_method: int = Field(default=1, ge=1)
    cache: bool = True
    tracking_uri: str | None = None

    @model_validator(mode="after")
    def methods_and_data(self) -> Self:
        if not self.methods:
            raise ValueError("At least one method is required")
        for name, method in self.methods.items():
            if not name or not name.replace("-", "_").isidentifier():
                raise ValueError(f"Invalid method name: {name!r}")
            if isinstance(method, Tabular) and self.dataset is None:
                raise ValueError(f"Method {name} requires a dataset")
            if isinstance(method, Lightning) and not method.predict_on:
                raise ValueError(f"Method {name} requires at least one image split")
            if isinstance(method, Lightning) and any(
                not split.replace("-", "_").isidentifier() for split in method.predict_on
            ):
                raise ValueError("Image split names must be identifiers")
        return self
