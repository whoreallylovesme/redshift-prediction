from lib.inference import inference
from lib.utils.hydra_utils import run_recipe_cli


def main() -> None:
    run_recipe_cli(lambda path: inference(config_path=path))


if __name__ == "__main__":
    main()
