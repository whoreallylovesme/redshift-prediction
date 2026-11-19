import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
from prefect.testing.utilities import prefect_test_harness

from lib.config.experiment import Experiment
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
                            "recipe": predictions,
                            "predictor": "absent_optional_regression_plugin.inference:run",
                        },
                        "table": {
                            "kind": "tabular",
                            "recipe": "tests/fixtures/legacy/models/ridge.yaml",
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
