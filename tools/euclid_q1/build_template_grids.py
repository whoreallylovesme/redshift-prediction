from __future__ import annotations

from pathlib import Path

import click

from lib.dataloaders.euclid_translation import TemplateGridConfig, build_template_grids
from lib.utils.utils import load_yaml_as


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
def main(config_path: Path) -> None:
    build_template_grids(load_yaml_as(str(config_path), TemplateGridConfig))


if __name__ == "__main__":
    main()
