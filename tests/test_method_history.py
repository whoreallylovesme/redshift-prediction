import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from lib.config.experiment import External, PredictionColumns, Saved
from lib.report.evaluation import load_history
from lib.workflow.experiment import report_record


class MethodHistoryTests(TestCase):
    def test_external_history_explicit_override_and_saved_isolation(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            predictions = directory / "predictions.parquet"
            predictions.touch()
            expected = [{"epoch": 1, "FVALID_logMSE": 0.07}]
            (directory / "train_report.json").write_text(json.dumps({"history": expected}))
            external = External(
                kind="external", predictor="fixtures:predict", recipe=Path("r.yaml")
            )
            record = report_record("table", "val", external, str(predictions), {})
            self.assertEqual(load_history(record), expected)

            staged = directory / "log.jsonl"
            staged.write_text('{"stage":"A","epoch":1,"val":{"margin":0.5}}\n')
            external.report = PredictionColumns(history=staged)
            record = report_record("staged", "val", external, str(predictions), {})
            self.assertEqual(load_history(record)[0]["val.margin"], 0.5)

            saved = Saved(kind="saved", predictions=predictions)
            self.assertNotIn("history", report_record("saved", "val", saved, str(predictions), {}))
