from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

from lib.config import FlowConfig
from photoz import legacy_flow as legacy
from runtime.split_tabular import prepare


class LegacyReportTests(TestCase):
    def test_tabular_and_nn_tracking_matches_export_without_retraining(self) -> None:
        with TemporaryDirectory() as temp, ExitStack() as stack:
            root = Path(temp)
            recipe = root / "recipe.yaml"
            recipe.write_text("seed: 2\n")
            tracker = MagicMock()
            tracker.start.side_effect = lambda name, params, flow_id: "run-" + name
            stack.enter_context(patch.object(legacy, "Tracker", return_value=tracker))
            stack.enter_context(patch.object(legacy, "split_tabular_data"))
            fit = stack.enter_context(patch.object(legacy, "fit_model"))
            stack.enter_context(patch.object(legacy, "combined_report", legacy.combined_report.fn))
            stack.enter_context(patch.object(legacy, "evaluate", legacy.evaluate.fn))

            def produce(_: str, out: str, **kwargs: Any) -> None:
                frame = pd.DataFrame({"y_true": [0.1, 0.2], "y_pred": [0.11, 0.19]})
                key = kwargs.get("id_column")
                if key is not None:
                    frame[key] = ["091", "043"]
                Path(out).parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(out)

            stack.enter_context(patch.object(legacy, "predict", side_effect=produce))
            uri = "sqlite:///tracking.db"
            for kind in ["tabular", "nn"]:
                directory = root / kind
                cfg = FlowConfig.model_validate(
                    {
                        "runs_root": str(directory),
                        "pipeline": kind,
                        "prepared_root": "unused",
                        "dataset": {
                            "input_csv": "unused",
                            "data_config": "unused",
                            "features": "unused",
                        },
                        "models": [{"name": "demo", "config": str(recipe), "feature_set": "set2"}],
                        "splits": [{"name": "B", "data_dir": "data", "catalog": "catalog"}],
                        "summaries": {"enabled": False},
                    }
                )
                run = legacy.run_tabular_pipeline if kind == "tabular" else legacy.run_nn_pipeline
                run(cfg, uri)
                manifest = json.loads((directory / "report-data.json").read_text())
                row = manifest["rows"][0]
                run_id = manifest["metadata"]["mlflow_runs"][row["name"]]
                self.assertEqual(fit.call_args.kwargs["run_id"], run_id)
                self.assertEqual(fit.call_args.kwargs["tracking_uri"], uri)
                tracker.log_metrics.assert_called_with(run_id, row["split"], row["metrics"])
                self.assertEqual(tracker.finish.call_args.args[:2], (run_id, True))
                self.assertEqual(row["paired_id"], kind == "nn")
                self.assertTrue((directory / "report.html").is_file())
                self.assertEqual(row["coverage"]["evaluated"], 2)

    def test_tracking_marks_training_failure_and_capsnet_config_is_effective(self) -> None:
        tracker = MagicMock()
        tracker.start.return_value = "rid"
        with patch.object(legacy, "Tracker", return_value=tracker):
            with self.assertRaisesRegex(RuntimeError, "failed fit"):
                with legacy.tracked_method("uri", "model", "recipe.yaml", Path("out")):
                    raise RuntimeError("failed fit")
            tracker.finish.assert_called_once_with("rid", False, Path("out"))
        for name in ("capsnet_hayat", "capsnet_teddy"):
            cfg = FlowConfig.model_validate(
                yaml.safe_load(Path(f"configs/flows/{name}.yaml").read_text())
            )
            self.assertIsNotNone(cfg.train)
            recipe = yaml.safe_load(Path((cfg.train or {})["config"]).read_text())
            self.assertEqual(set(cfg.train or {}), {"config"})
            self.assertTrue(Path(recipe["out_dir"]).is_relative_to(cfg.runs_root))
            self.assertTrue("out" in recipe["inference"] or "splits" in recipe["inference"])

    def test_split_retains_configured_exact_string_ids(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "table.csv"
            path.write_text("id,x,target\n001,1,1\n020,2,2\n003,3,3\n031,4,4\n")
            cfg = root / "data.yaml"
            cfg.write_text("split: {test_size: 0.5, shuffle: false}\n")
            features = root / "features.yaml"
            features.write_text("target: target\nfeature_sets: {set2: [x]}\n")
            prepare(str(path), str(cfg), str(features), str(root), "id")
            self.assertEqual(pd.read_parquet(root / "test.parquet").id.tolist(), ["003", "031"])

    def test_native_missing_predictions_remain_visible_but_unscored(self) -> None:
        from lib.report import build_report_data
        from photoz.report import PHOTOZ
        from runtime.evaluate import calc_metrics

        with TemporaryDirectory() as directory:
            out = Path(directory)
            preds = out / "preds.parquet"
            pd.DataFrame(
                {
                    "object_id": [1, 2, 3, 4],
                    "y_true": [0.1, 0.2, 0.3, None],
                    "y_pred": [0.4, None, 99, 0.4],
                    "status": ["ok", "unreadable", "rejected", "ok"],
                }
            ).to_parquet(preds)
            calc_metrics(str(preds), str(out), "CNN")
            legacy_report = json.loads((out / "report.json").read_text())["test_metrics"]
            row = build_report_data(
                [{"name": "CNN", "split": "test", "predictions": preds}], PHOTOZ, "x"
            )["rows"][0]
            self.assertEqual(row["metrics"], legacy_report)
            self.assertEqual(row["coverage"]["evaluated"], 1)
            self.assertEqual(row["coverage"]["requested"], 4)
