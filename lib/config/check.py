from pathlib import Path
from typing import Any

import yaml

from lib.config import FeaturesConfig, FlowConfig, PytorchTrainConfig, SklearnTrainConfig
from lib.config.experiment import Experiment, External, Lightning, Saved, Tabular
from lib.models.factories import FACTORIES
from lib.workflow.data import symbol


def mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return raw


def check_recipe(path: Path) -> None:
    raw = mapping(path)
    if "pipeline" in raw:
        legacy = FlowConfig.model_validate(raw)
        paths = [m["config"] for m in legacy.models]
        if legacy.dataset:
            paths.extend(
                legacy.dataset[k] for k in ("features", "data_config") if k in legacy.dataset
            )
        if legacy.train:
            paths.append(legacy.train["config"])
        if legacy.euclid:
            euclid = legacy.euclid
            paths.extend(
                stage.config
                for stage in (euclid.pretrain, euclid.finetune, euclid.classical, euclid.report)
                if stage is not None
            )
        for name in paths:
            mapping(Path(name))
        return

    exp = Experiment.model_validate(raw)
    if exp.task != "regression":
        from lib.report.task import ReportTask

        if not isinstance(symbol(exp.task), ReportTask):
            raise ValueError(f"Expected a ReportTask at {exp.task}")
    if exp.dataset:
        ds = exp.dataset
        for entry in (ds.reader, ds.cleaner):
            if entry and not callable(symbol(entry)):
                raise ValueError(f"Expected a callable at {entry}")
        if ds.feature_catalog:
            features = FeaturesConfig.model_validate(mapping(ds.feature_catalog))
            if ds.feature_set not in features.feature_sets:
                raise ValueError(f"Unknown feature set {ds.feature_set} in {ds.feature_catalog}")
    for method in exp.methods.values():
        if isinstance(method, Saved):
            continue
        if isinstance(method, External):
            if not method.recipe.is_file():
                raise FileNotFoundError(method.recipe)
            if not callable(symbol(method.predictor)):
                raise ValueError(f"Expected a predictor at {method.predictor}")
            continue
        recipe = mapping(method.recipe)
        if isinstance(method, Tabular):
            estimator = SklearnTrainConfig.model_validate(recipe).estimator
        elif isinstance(method, Lightning):
            model = PytorchTrainConfig.model_validate(recipe)
            estimator = model.estimator
            if model.training_task and not callable(symbol(model.training_task)):
                raise ValueError(f"Expected a training task at {model.training_task}")
        if estimator not in FACTORIES and "estimator =" not in estimator:
            raise ValueError(f"Unknown estimator {estimator!r} in {method.recipe}")
