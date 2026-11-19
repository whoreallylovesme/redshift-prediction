from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict

from lib.config.experiment import Experiment as Experiment
from lib.config.schemas import PytorchTrainConfig as PytorchTrainConfig
from lib.config.schemas import SklearnTrainConfig as SklearnTrainConfig


def load_flow(
    path: Path,
    *,
    check: bool = False,
    overrides: tuple[str, ...] = (),
    workers: int | None = None,
    tracking_uri: str | None = None,
) -> Experiment:
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
    return Experiment.model_validate(raw, context={"check": check})
