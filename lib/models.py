from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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


def _make_scaler(cfg: ExperimentConfig):
    if not cfg.scaler:
        return None

    scaler_type = cfg.scaler.get("type")
    scaler_params = cfg.scaler.get("params", {}) or {}

    if scaler_type == "standard":
        return StandardScaler(**scaler_params)

    if scaler_type == "minmax":
        return MinMaxScaler(**scaler_params)

    if scaler_type == "robust":
        return RobustScaler(**scaler_params)

    raise ValueError(f"Unknown scaler type: {scaler_type}")


def make_pipeline(cfg: ExperimentConfig) -> Pipeline:
    base_estimator = _make_base_estimator(cfg)
    steps = []

    scaler = _make_scaler(cfg)
    if scaler is not None:
        steps.append(("scaler", scaler))

    steps.append(("model", base_estimator))

    return Pipeline(steps)
