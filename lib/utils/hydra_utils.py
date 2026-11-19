from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from hydra import main as hydra_main
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf


def _recipe_path(value: str, config_dir: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    roots = [config_dir] if config_dir else []
    if HydraConfig.initialized():
        roots.extend(
            Path(source.path)
            for source in HydraConfig.get().runtime.config_sources
            if source.schema == "file"
        )
    roots.append(Path.cwd() / "configs")
    for root in roots:
        relative = Path(*path.parts[1:]) if path.parts[0] == root.name else path
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return path


def hydra_to_yaml(cfg: DictConfig, config_dir: Path | None = None) -> Path:
    values = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(values, dict):
        raise ValueError("Expected a Hydra mapping")
    groups: dict[str, Any] = {str(key): value for key, value in values.items()}
    if isinstance(groups.get("model"), dict):
        model = groups.pop("model")
        dataset = groups.pop("dataset", {})
        train = groups.pop("train", {})
        if "recipe" in model:
            path = _recipe_path(model["recipe"], config_dir)
            base = OmegaConf.load(path)
            if not isinstance(base, DictConfig):
                raise ValueError("Referenced model recipe must be a mapping")
            if "h5_path" in dataset and "data_dir" in base and "h5_path" not in base:
                raise ValueError(f"{path.name} expects an image directory, not an HDF5 dataset")
            overrides = {key: value for key, value in train.items() if key in base.train}
            if "seed" in groups:
                overrides["seed"] = groups.pop("seed")
            inputs = {key: value for key, value in dataset.items() if key in base}
            result = OmegaConf.merge(base, {"train": overrides}, inputs, groups)
            assert isinstance(result, DictConfig)
            if not result.get("out_dir"):
                result["out_dir"] = f"nn_runs/{path.stem}"
        else:
            result = OmegaConf.create({**groups, **train, **model, **dataset})
    else:
        result = OmegaConf.create(groups)
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as stream:
        OmegaConf.save(result, stream)
        return Path(stream.name)


def run_recipe_cli(action: Callable[[str], None]) -> None:
    argv = sys.argv[1:]
    if any(arg == "--config" or arg.startswith("--config=") for arg in argv):
        parser = argparse.ArgumentParser(description="Run one resolved YAML recipe")
        parser.add_argument("--config", type=Path, required=True)
        action(str(parser.parse_args(argv).config))
        return

    # An appended Hydra search path has lower priority than the main directory.
    # Set the external directory as the sole source of the user's config name.
    external = any(arg.split("=")[0] in {"--config-dir", "-cd"} for arg in argv)
    # Only archived experiments use the grouped single-stage CLI.
    default = next(
        (
            p.resolve()
            for p in (Path("configs"), Path("examples/configs/hydra"))
            if (p / "hydra.yaml").is_file()
        ),
        None,
    )
    if not external and default is None:
        parser = argparse.ArgumentParser(description="Debug a single stage of a saved experiment.")
        parser.add_argument("--config", help="Resolved method YAML from the run's config_snapshot")
        parser.add_argument("--config-dir", help="Directory containing archived Hydra groups")
        if not argv or argv in (["--help"], ["-h"]):
            parser.print_help()
            return
        parser.error("pass --config or --config-dir to select an experiment")

    @hydra_main(
        config_path=None if external else str(default), config_name="hydra", version_base="1.3"
    )
    def composed(cfg: DictConfig) -> None:
        path = hydra_to_yaml(cfg)
        try:
            action(str(path))
        finally:
            path.unlink(missing_ok=True)

    composed()
