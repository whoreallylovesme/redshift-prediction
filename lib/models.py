from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

from .config import ExperimentConfig
from .llr import LocalLinearRegression


def _make_base_estimator(cfg: ExperimentConfig):
    if cfg.model_type == "catboost":
        return CatBoostRegressor(**cfg.params)

    if cfg.model_type == "local_linear":
        return LocalLinearRegression(**cfg.params)

    if cfg.model_type == "mlp":
        return MLPRegressor(**cfg.params)

    raise ValueError(f"Unknown model_type: {cfg.model_type}")


def make_pipeline(cfg: ExperimentConfig) -> Pipeline:
    base_estimator = _make_base_estimator(cfg)

    pipe = Pipeline([
        ("model", base_estimator),
    ])

    return pipe
