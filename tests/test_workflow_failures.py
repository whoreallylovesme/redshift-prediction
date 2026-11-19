import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
from prefect.testing.utilities import prefect_test_harness
from pydantic import ValidationError

from lib.config import PytorchTrainConfig
from lib.config.experiment import Experiment
from lib.config.schemas import ImageCatalogConfig
from lib.workflow.data import code_version
from lib.workflow.experiment import run


class WorkflowFailuresTest(TestCase):
    def test_plugin_fingerprint_tracks_single_module_contents_and_missing_modules(self) -> None:
        name = "temporary_workflow_plugin"
        with TemporaryDirectory() as tmp, patch.object(sys, "path", [tmp, *sys.path]):
            module = Path(tmp) / f"{name}.py"
            missing = code_version([f"{name}:predict"])
            module.write_text("def predict(): return 1\n")
            first = code_version([f"{name}:predict"])
            module.write_text("def predict(): return 2\n")
            second = code_version([f"{name}:predict"])
            self.assertEqual(len({missing, first, second}), 3)

    def test_missing_external_package_preserves_independent_saved_report(self) -> None:
        with TemporaryDirectory() as tmp, prefect_test_harness():
            root = Path(tmp)
            predictions = root / "predictions.parquet"
            pd.DataFrame(
                {"object_id": [1, 2], "y_true": [1.0, 2.0], "y_pred": [1.0, 2.0]}
            ).to_parquet(predictions)
            cfg = Experiment.model_validate(
                {
                    "name": "independent methods",
                    "output": root / "runs",
                    "dataset": {"source": root / "missing.csv", "target": "y", "features": ["x"]},
                    "methods": {
                        "published": {"kind": "saved", "predictions": predictions},
                        "absent": {
                            "kind": "external",
                            "model": "absent_optional_regression_plugin.inference:run",
                        },
                        "table": {
                            "kind": "tabular",
                            "model": "ridge",
                        },
                    },
                }
            )
            with self.assertRaisesRegex(RuntimeError, "Failed methods: absent"):
                run(cfg)
            [report_path] = (cfg.output).glob("*/report-data.json")
            report = json.loads(report_path.read_text())
            rows = {row["name"]: row for row in report["rows"]}
            self.assertEqual(rows["absent"]["status"], "failed")
            self.assertIn("absent_optional_regression_plugin", rows["absent"]["details"]["reason"])
            self.assertEqual(rows["table"]["status"], "failed")
            self.assertIn("missing.csv", rows["table"]["details"]["reason"])
            self.assertIn("missing.csv", report["metadata"]["data"]["error"])
            self.assertEqual(rows["published"]["status"], "completed")
            self.assertEqual(rows["published"]["metrics"]["mae"], 0.0)
            self.assertTrue((report_path.parent / "report.html").is_file())


    def test_neural_configs_reject_unknown_fields(self) -> None:
        with self.assertRaises(ValidationError):
            ImageCatalogConfig.model_validate(
                {"id_column": "id", "target_column": "y", "unknown": True}
            )
        with self.assertRaises(ValidationError):
            PytorchTrainConfig.model_validate({
                "model_type": "lightning",
                "estimator": "capsnet",
                "data_dir": ".",
                "catalog": ".",
                "image_catalog": {"id_column": "id", "target_column": "y"},
                "epochs": 1,
                "batch_size": 1,
                "lr": 0.1,
                "unknown": True,
            })


    def test_preparation_graph_rejects_cycles(self) -> None:
        with self.assertRaisesRegex(ValidationError, "cycle"):
            Experiment.model_validate({
                "name": "cycle",
                "output": ".",
                "preparation": {
                    "a": {
                        "action": "example:action",
                        "needs": ["b"],
                        "outputs": ["a"],
                    },
                    "b": {
                        "action": "example:action",
                        "needs": ["a"],
                        "outputs": ["b"],
                    },
                },
                "methods": {"saved": {"kind": "saved", "predictions": "predictions.parquet"}},
            })
