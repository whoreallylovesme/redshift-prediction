from __future__ import annotations

import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

_active: ContextVar[tuple[str, str] | None] = ContextVar("mlflow_run", default=None)


@contextmanager
def bind_run(uri: str | None, run_id: str | None) -> Iterator[None]:
    token = _active.set((uri, run_id) if uri and run_id else None)
    try:
        yield
    finally:
        _active.reset(token)


def lightning_logger(default: Any = False) -> Any:
    active = _active.get()
    if active is None:
        return default
    from lightning.pytorch.loggers import MLFlowLogger

    uri, run_id = active
    return MLFlowLogger(tracking_uri=uri, run_id=run_id, log_model=False)


class Tracker:
    def __init__(self, uri: str, experiment: str) -> None:
        try:
            from mlflow import MlflowClient
        except ImportError as exc:
            raise RuntimeError("Install tracking support with uv sync --extra tracking") from exc

        self.client = MlflowClient(tracking_uri=uri)
        info = self.client.get_experiment_by_name(experiment)
        if info is None:
            from mlflow.exceptions import MlflowException

            try:
                artifact_location = None
                if uri.startswith("sqlite:///"):
                    database = Path(uri.removeprefix("sqlite:///").split("?", 1)[0]).resolve()
                    artifact_location = str(database.parent / "mlartifacts")
                self.experiment_id = self.client.create_experiment(
                    experiment, artifact_location=artifact_location
                )
            except MlflowException:
                info = self.client.get_experiment_by_name(experiment)
                if info is None:
                    raise
                self.experiment_id = info.experiment_id
        else:
            self.experiment_id = info.experiment_id

    def start(self, name: str, params: Mapping[str, Any], flow_id: str) -> str:
        prior = self.client.search_runs(
            [self.experiment_id],
            filter_string=f"tags.`prefect.flow_run_id` = '{flow_id}' AND tags.method = '{name}'",
            max_results=1,
        )
        if prior:
            run_id: str = prior[0].info.run_id
            self.client.update_run(run_id, status="RUNNING")
        else:
            run = self.client.create_run(
                self.experiment_id,
                run_name=name,
                tags={"prefect.flow_run_id": flow_id, "method": name},
            )
            run_id = run.info.run_id
        try:
            for key, value in params.items():
                self.client.log_param(run_id, key, json.dumps(value, default=str)[:6000])
        except Exception:
            self.client.set_terminated(run_id, status="FAILED")
            raise
        return run_id

    def log_metrics(self, run_id: str, split: str, metrics: Mapping[str, Any]) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.client.log_metric(run_id, f"{split}/{key}", float(value))

    def log_snapshot(self, run_id: str, snapshot: Path, attempt: int = 1) -> None:
        path = "config_snapshot" if attempt <= 1 else f"attempt-{attempt}/config_snapshot"
        self.client.log_artifacts(run_id, str(snapshot), path)
        self.client.set_tag(run_id, "config_snapshot", path)

    def finish(self, run_id: str, success: bool, directory: Path) -> None:
        try:
            self.client.set_tag(run_id, "output_directory", str(directory.absolute()))
        finally:
            self.client.set_terminated(run_id, status="FINISHED" if success else "FAILED")


def finish_with_snapshots(
    tracker: Tracker | None, snapshot: Path, output: Path, attempt: int, runs: dict[str, bool]
) -> None:
    from lib.config.snapshot import seal_snapshot

    active = sys.exception()
    errors: list[Exception] = []
    try:
        seal_snapshot(snapshot)
        sealed = True
    except Exception as exc:
        errors.append(exc)
        sealed = False
    if tracker:
        for run_id, success in runs.items():
            if sealed:
                try:
                    tracker.log_snapshot(run_id, snapshot, attempt)
                except Exception as exc:
                    errors.append(exc)
                    success = False
            try:
                tracker.finish(run_id, success and sealed, output)
            except Exception as exc:
                errors.append(exc)
    if errors:
        if active is not None:
            active.add_note(f"Configuration snapshot or tracking cleanup failed: {errors!r}")
        else:
            raise ExceptionGroup("Configuration snapshot or tracking cleanup failed", errors)
