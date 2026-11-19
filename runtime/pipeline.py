from pathlib import Path

import click
from hydra.errors import HydraException
from omegaconf.errors import OmegaConfBaseException

from lib.config import load_flow


def pipeline(
    config_path: str,
    tracking_uri: str | None = None,
    workers: int | None = None,
) -> Path:
    from lib.workflow.experiment import run

    return run(load_flow(Path(config_path), workers=workers, tracking_uri=tracking_uri))


@click.command(no_args_is_help=True)
@click.argument("overrides", nargs=-1, metavar="[KEY=VALUE]...")
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--render",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Rebuild the offline report from its saved data.",
)
@click.option("--check", is_flag=True, help="Validate flow, recipes and plugin references.")
@click.option("--tracking-uri", envvar="MLFLOW_TRACKING_URI", help="MLflow Tracking URI.")
@click.option("--workers", type=click.IntRange(min=1), help="Maximum concurrent method tasks.")
def main(
    overrides: tuple[str, ...],
    config_path: str | None,
    render: Path | None,
    check: bool,
    tracking_uri: str | None,
    workers: int | None,
) -> None:
    if render:
        if config_path or check or overrides:
            raise click.UsageError("Choose --render or --config")
        from lib.workflow.experiment import rebuild_report

        rebuild_report(render)
        click.echo(render / "report.html")
        return
    if config_path is None:
        raise click.UsageError("Pass --config for an experiment or --render for a saved report.")
    try:
        config = load_flow(
            Path(config_path), check=check, overrides=overrides, workers=workers,
            tracking_uri=tracking_uri,
        )
    except (ValueError, OSError, HydraException, OmegaConfBaseException) as exc:
        raise click.ClickException(str(exc)) from exc
    if check:
        click.echo(f"OK {config_path}")
        return
    from lib.workflow.experiment import run

    click.echo(run(config) / "report.html")


if __name__ == "__main__":
    main()
