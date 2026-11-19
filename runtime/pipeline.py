from pathlib import Path
from typing import Any

import click
from hydra.errors import HydraException
from omegaconf.errors import OmegaConfBaseException

from lib.config import Experiment, FlowConfig, load_flow

_LEGACY_STAGES = (
    "derived_config",
    "evaluate",
    "evaluate_capsnet_split",
    "fit_capsnet",
    "fit_euclid_stage",
    "fit_model",
    "predict",
    "predict_capsnet_splits",
    "predict_euclid_stage",
    "render_summary_plots",
    "render_summary_table",
    "report_euclid_stage",
    "run_capsnet_pipeline",
    "run_euclid_classical",
    "run_euclid_clip_pipeline",
    "run_euclid_finetune",
    "run_euclid_inference",
    "run_euclid_pretrain",
    "run_euclid_report",
    "run_nn_pipeline",
    "run_tabular_pipeline",
    "split_tabular_data",
)
__all__ = ["pipeline", "main", *_LEGACY_STAGES]


def __getattr__(name: str) -> Any:
    if name in _LEGACY_STAGES:
        from photoz import legacy_flow

        return getattr(legacy_flow, name)
    raise AttributeError(name)


def pipeline(
    config_path: str,
    tracking_uri: str | None = None,
    workers: int | None = None,
) -> Path | None:
    config = load_flow(Path(config_path), workers=workers, tracking_uri=tracking_uri)
    return _run(config)


def _run(config: Experiment | FlowConfig) -> Path | None:
    if isinstance(config, FlowConfig):
        from photoz.legacy_flow import legacy_pipeline

        legacy_pipeline(config)
        return None
    from lib.workflow.experiment import run

    return run(config)


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
    directory = _run(config)
    if directory:
        click.echo(directory / "report.html")


if __name__ == "__main__":
    main()
