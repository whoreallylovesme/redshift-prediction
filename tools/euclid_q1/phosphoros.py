from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from photoz.methods import phosphoros
from photoz.methods.phosphoros import (
    SIDES,
    build_grids,
    build_south_workspace,
    chosen_sets,
    load_config,
    postprocess_set,
    prepare_set,
    run_set,
)


def __getattr__(name: str) -> Any:
    return getattr(phosphoros, name)


CONFIG = click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True
)
SETS = click.option("--set", "names", multiple=True)
VARIANT = click.option("--variant", default=None)


@click.group()
def cli() -> None:
    pass


@cli.command("workspace")
@CONFIG
def workspace_command(config_path: Path) -> None:
    build_south_workspace(load_config(config_path))


@cli.command("grids")
@CONFIG
@click.option("--side", "side_name", type=click.Choice(sorted(SIDES)), required=True)
def grids_command(config_path: Path, side_name: str) -> None:
    build_grids(load_config(config_path), side_name)


@cli.command("prepare")
@CONFIG
@SETS
def prepare_command(config_path: Path, names: tuple[str, ...]) -> None:
    cfg = load_config(config_path)
    for name in chosen_sets(cfg, names, None):
        prepare_set(cfg, name)


@cli.command("run")
@CONFIG
@SETS
@VARIANT
def run_command(config_path: Path, names: tuple[str, ...], variant: str | None) -> None:
    cfg = load_config(config_path)
    for name in chosen_sets(cfg, names, variant):
        run_set(cfg, name, variant)


@cli.command("postprocess")
@CONFIG
@SETS
@VARIANT
def postprocess_command(config_path: Path, names: tuple[str, ...], variant: str | None) -> None:
    cfg = load_config(config_path)
    for name in chosen_sets(cfg, names, variant):
        postprocess_set(cfg, name, variant)


if __name__ == "__main__":
    cli()
