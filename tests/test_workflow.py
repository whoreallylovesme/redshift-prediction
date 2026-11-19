from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from typing import Any
from unittest import TestCase

import pandas as pd
from prefect.client.orchestration import get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterId
from prefect.testing.utilities import prefect_test_harness
from pydantic import ValidationError

from lib.config.experiment import Experiment
from lib.workflow.experiment import rebuild_report, run

rendezvous = Barrier(2)


def simultaneous(config_path: Path, out_dir: Path) -> Path:
    rendezvous.wait(timeout=15)
    return config_path


class WorkflowTest(TestCase):
    def test_unrecognized_config_fields_fail(self) -> None:
        with self.assertRaises(ValidationError):
            Experiment.model_validate({"name": "x", "output": ".", "workerss": 2, "methods": {}})

    def test_prefect_cache_parallelism_and_sqlite_tracking(self) -> None:
        from importlib.util import find_spec
        from uuid import UUID

        with TemporaryDirectory() as tmp, prefect_test_harness():
            root = Path(tmp)
            source = root / "data.parquet"
            frame = pd.DataFrame(
                {
                    "id": range(24),
                    "x": range(24),
                    "y": [i * 0.75 for i in range(24)],
                    "role": ["test" if i % 4 == 0 else "train" for i in range(24)],
                }
            )
            frame.to_parquet(source)
            cfg: dict[str, Any] = {
                "name": "workflow-test",
                "output": str(root / "runs"),
                "workers": 2,
                "dataset": {
                    "source": str(source),
                    "id_column": "id",
                    "features": ["x"],
                    "target": "y",
                    "split_column": "role",
                },
                "methods": {
                    "ridge": {"kind": "tabular", "recipe": "examples/configs/models/ridge.yaml"}
                },
            }
            if find_spec("mlflow"):
                cfg["tracking_uri"] = f"sqlite:///{root / 'tracking.db'}"
            config = Experiment.model_validate(cfg)

            def fit_state(directory: Path) -> str | None:
                with get_client(sync_client=True) as client:
                    tasks = client.read_task_runs(
                        flow_run_filter=FlowRunFilter(
                            id=FlowRunFilterId(any_=[UUID(directory.name)]),
                        )
                    )
                    return next(t.state_name for t in tasks if t.name == "ridge-fit")

            first = run(config)
            second = run(config)
            self.assertEqual(fit_state(first), "Completed")
            self.assertEqual(fit_state(second), "Cached")
            pred = second / "methods/ridge/predictions.parquet"
            self.assertEqual(pd.read_parquet(pred)["id"].tolist(), list(range(0, 24, 4)))
            metrics_path = second / "metrics.json"
            metrics = json.loads(metrics_path.read_text())
            if config.tracking_uri:
                from mlflow import MlflowClient

                ids = json.loads((second / "tracking.json").read_text())
                tracked = MlflowClient(tracking_uri=config.tracking_uri).get_run(ids["ridge"])
                self.assertEqual(tracked.info.status, "FINISHED")
                self.assertEqual(
                    tracked.data.metrics["test/mae"], metrics["rows"][0]["metrics"]["mae"]
                )
            frame.loc[1, "y"] += 1
            frame.to_parquet(source)
            self.assertEqual(fit_state(run(config)), "Completed")
            config_file = str(pred)
            method = {
                "kind": "external",
                "predictor": f"{__name__}:simultaneous",
                "recipe": config_file,
                "report": {"id_column": "id"},
            }
            parallel = Experiment.model_validate(
                {
                    "name": "parallel",
                    "output": str(root / "parallel"),
                    "workers": 2,
                    "methods": {"first": method, "second": method},
                }
            )
            both = run(parallel)
            self.assertEqual(len(json.loads((both / "metrics.json").read_text())["rows"]), 2)
            pred.unlink()
            rebuild_report(second)
            self.assertEqual(json.loads(metrics_path.read_text()), metrics)
