from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from lib.config.experiment import Experiment as Experiment
from lib.config.schemas import DatasetConfig as DatasetConfig
from lib.config.schemas import EuclidClassicalStage as EuclidClassicalStage
from lib.config.schemas import EuclidFinetuneStage as EuclidFinetuneStage
from lib.config.schemas import EuclidFlowConfig as EuclidFlowConfig
from lib.config.schemas import EuclidPretrainStage as EuclidPretrainStage
from lib.config.schemas import EuclidStage as EuclidStage
from lib.config.schemas import FeaturesConfig as FeaturesConfig
from lib.config.schemas import FlowConfig as FlowConfig
from lib.config.schemas import PytorchTrainConfig as PytorchTrainConfig
from lib.config.schemas import SklearnTrainConfig as SklearnTrainConfig


def load_flow(
    path: Path,
    *,
    check: bool = False,
    overrides: tuple[str, ...] = (),
    workers: int | None = None,
    tracking_uri: str | None = None,
) -> Experiment | FlowConfig:
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"{path} must contain a flow mapping")
    if "defaults" in config or overrides:
        if path.suffix != ".yaml":
            raise ValueError("Use a .yaml file for Hydra config groups or command-line overrides")
        with initialize_config_dir(config_dir=str(path.resolve().parent), version_base="1.3"):
            config = compose(config_name=path.name, overrides=list(overrides))
    with open_dict(config):
        if workers is not None:
            config.workers = workers
        if tracking_uri:
            config.tracking_uri = tracking_uri
    raw = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    assert isinstance(raw, dict)
    if "pipeline" in raw:
        if raw.get("workers", 1) != 1:
            raise ValueError(
                "Historical flows use serial execution; configure workers in a methods flow"
            )
        return FlowConfig.model_validate(raw, context={"check": check})
    return Experiment.model_validate(raw, context={"check": check})
