from hydra import main as hydra_main
from omegaconf import DictConfig

from lib.hydra_utils import hydra_to_yaml
from lib.inference import inference


@hydra_main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    config_path = hydra_to_yaml(cfg)
    inference(config_path=str(config_path))


if __name__ == "__main__":
    main()
