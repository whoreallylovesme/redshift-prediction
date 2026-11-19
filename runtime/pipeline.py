from pathlib import Path
from typing import Any

import click
import yaml

from lib.config.experiment import Experiment

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
    raw = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("A flow recipe must be a YAML mapping")
    if tracking_uri:
        raw["tracking_uri"] = tracking_uri
    if workers is not None:
        raw["workers"] = workers
    if "pipeline" in raw:
        from photoz.legacy_flow import legacy_pipeline

        legacy_pipeline(config_path, tracking_uri=tracking_uri, workers=workers)
        return None
    from lib.workflow.experiment import run

    return run(Experiment.model_validate(raw))


@click.command()
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
    config_path: str | None,
    render: Path | None,
    check: bool,
    tracking_uri: str | None,
    workers: int | None,
) -> None:
    if render:
        if config_path or check:
            raise click.UsageError("Choose --render or --config")
        from lib.workflow.experiment import rebuild_report

        rebuild_report(render)
        click.echo(render / "report.html")
        return
    path = config_path or "configs/flows/run_all.yaml"
    if check:
        from lib.config.check import check_recipe

        check_recipe(Path(path))
        click.echo(f"OK {path}")
        return
    directory = pipeline(path, tracking_uri, workers)
    if directory:
        click.echo(directory / "report.html")


if __name__ == "__main__":
    main()
