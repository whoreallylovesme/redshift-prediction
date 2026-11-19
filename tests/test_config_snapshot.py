from __future__ import annotations

import json
from contextlib import chdir
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import pandas as pd
import yaml
from prefect.testing.utilities import prefect_test_harness

from lib.config import load_flow
from lib.config.experiment import Experiment
from lib.config.snapshot import (
    create_snapshot,
    record_config,
    recording_configs,
    restore_snapshot,
    seal_snapshot,
)
from lib.tracking import Tracker, finish_with_snapshots
from photoz.legacy_flow import derived_config
from photoz.methods.phosphoros import VariantConfig, config_values, variant_factors
from runtime.pipeline import pipeline
from tests.test_legacy_pipeline_e2e import table_flow


def use_saved_predictions(config_path: Path, out_dir: Path) -> Path:
    return config_path


class SnapshotTests(TestCase):
    def test_replay_fields_privacy_and_only_used_native_inputs(self) -> None:
        with TemporaryDirectory() as temp, chdir(temp):
            root = Path(temp).resolve()
            Path("cfg").mkdir()
            cfg = Path("cfg/recipe.yml")
            cfg.write_text("type: SED\ntracking_uri: https://sensitive.invalid/token\n")
            dust = Path("cfg/active.conf")
            dust.write_text("catalog-type=synthetic\n")
            Path("cfg/inactive.conf").write_text("unused=yes\n")
            factors = Path("cfg/calibration.json")
            factors.write_text('{"VIS": 1.5}')
            Path("objects.parquet").write_bytes(b"catalog contents")
            Path("split.yaml").write_text("split: {test_size: 0.2}")
            exp = Experiment.model_validate(
                {
                    "name": "external",
                    "output": "runs",
                    "tracking_uri": "https://sensitive.invalid/token",
                    "workers": 2,
                    "dataset": {
                        "source": "objects.parquet",
                        "target": "y",
                        "features": ["x"],
                        "split_config": "split.yaml",
                    },
                    "methods": {
                        "native": {
                            "kind": "external",
                            "recipe": str(cfg),
                            "predictor": "example:predict",
                        },
                        "payload": {
                            "kind": "external",
                            "recipe": "objects.parquet",
                            "predictor": "example:predict",
                        },
                    },
                }
            )
            archive = create_snapshot(root / "archived", exp)
            with recording_configs(archive):
                invalid = Path("cfg/invalid.yaml")
                invalid.write_text("tracking_uri: [https://sensitive.invalid\n")
                self.assertIsNone(record_config(invalid))
                derived_config(str(cfg), Path("derived/stage.yaml"), seed=13)
                self.assertEqual(config_values(dust)["catalog-type"], "synthetic")
                self.assertEqual(
                    variant_factors(VariantConfig(sets=["x"], multiply_json=factors)),
                    ({}, {"VIS": 1.5}),
                )
            seal_snapshot(archive)
            manifest = json.loads((archive / "manifest.json").read_text())
            paths = set(manifest["sha256"])
            self.assertEqual(
                paths,
                {
                    "effective.json",
                    "flow.yaml",
                    "files/workspace/split.yaml",
                    "files/workspace/cfg/recipe.yml",
                    "files/workspace/cfg/active.conf",
                    "files/workspace/cfg/calibration.json",
                    "files/workspace/derived/stage.yaml",
                },
            )
            rebuilt = load_flow(archive / "flow.yaml")
            assert isinstance(rebuilt, Experiment)
            self.assertEqual(rebuilt.workers, 2)
            for file in archive.rglob("*"):
                if file.is_file():
                    self.assertNotIn(b"sensitive.invalid", file.read_bytes())
            restored = root / "restored"
            restore_snapshot(archive, restored)
            self.assertEqual((restored / dust).read_bytes(), dust.read_bytes())
            self.assertEqual((restored / factors).read_bytes(), factors.read_bytes())
            (restored / cfg).write_text("different")
            with self.assertRaises(FileExistsError):
                restore_snapshot(archive, restored)
            del manifest["sha256"]["flow.yaml"]
            (archive / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "missing the flow configuration"):
                restore_snapshot(archive, restored)

    def test_finalization_attempts_every_run_even_after_failure(self) -> None:
        tracker = MagicMock(spec=Tracker)
        source, output = Path("snapshot"), Path("work")
        with patch("lib.config.snapshot.seal_snapshot", side_effect=OSError("disk full")):
            with self.assertRaises(ExceptionGroup):
                finish_with_snapshots(tracker, source, output, 1, {"a": True, "b": True})
        tracker.log_snapshot.assert_not_called()
        self.assertEqual(
            tracker.finish.call_args_list, [call("a", False, output), call("b", False, output)]
        )
        tracker.reset_mock()
        tracker.log_snapshot.side_effect = [OSError("upload"), None]
        tracker.finish.side_effect = [OSError("unavailable"), None]
        with patch("lib.config.snapshot.seal_snapshot"):
            with self.assertRaises(ExceptionGroup) as e:
                finish_with_snapshots(tracker, source, output, 3, {"a": True, "b": True})
        self.assertEqual(len(e.exception.exceptions), 2)
        self.assertEqual(
            tracker.finish.call_args_list, [call("a", False, output), call("b", True, output)]
        )

    def test_public_flows_publish_config_and_replay_without_original_recipe(self) -> None:
        with TemporaryDirectory() as temp, prefect_test_harness(), chdir(temp):
            root = Path(temp).resolve()
            model = Path("models/ridge.yaml")
            model.parent.mkdir()
            model.write_text("estimator: ridge\n")
            pd.DataFrame(
                {"idx": list(range(12)), "x": list(range(12)), "y": list(range(12))}
            ).to_parquet("train.parquet")
            flow = Path("flow.yaml")
            flow.write_text(
                yaml.safe_dump(
                    {
                        "name": "snapshot-test",
                        "cache": False,
                        "output": "runs",
                        "dataset": {
                            "source": "train.parquet",
                            "id_column": "idx",
                            "target": "y",
                            "features": ["x"],
                        },
                        "methods": {"ridge": {"kind": "tabular", "recipe": str(model)}},
                    }
                )
            )
            uri = f"sqlite:///{root / 'mlflow.sqlite'}" if find_spec("mlflow") else None
            first = pipeline(str(flow), tracking_uri=uri)
            assert isinstance(first, Path)
            archive = first / "config_snapshot"
            original_predictions = pd.read_parquet(first / "methods/ridge/predictions.parquet")
            model.unlink()
            flow.unlink()
            replay = pipeline(str(restore_snapshot(archive, root)))
            assert isinstance(replay, Path)
            pd.testing.assert_frame_equal(
                original_predictions, pd.read_parquet(replay / "methods/ridge/predictions.parquet")
            )
            old_output, old_flow = table_flow(root)
            pipeline(old_flow, tracking_uri=uri)
            (legacy,) = (old_output / "config_snapshots").iterdir()
            self.assertEqual(
                yaml.safe_load((legacy / "flow.yaml").read_text())["pipeline"], "tabular"
            )
            partial = root / "partial.yml"
            partial.write_text(
                yaml.safe_dump(
                    {
                        "name": "snapshot-partial",
                        "output": "partial",
                        "methods": {
                            "unavailable": {
                                "kind": "external",
                                "predictor": "photoz.methods.phosphoros:predict",
                                "recipe": "absent.yaml",
                                "options": {"set_name": "example"},
                            },
                            "available": {
                                "kind": "external",
                                "predictor": f"{__name__}:use_saved_predictions",
                                "recipe": str(first / "methods/ridge/predictions.parquet"),
                            },
                        },
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "Failed methods: unavailable"):
                pipeline(str(partial), tracking_uri=uri)
            (mixed,) = (root / "partial").iterdir()
            states = json.loads((mixed / "report-data.json").read_text())["rows"]
            self.assertEqual(
                {r["name"]: r["status"] for r in states},
                {"unavailable": "failed", "available": "completed"},
            )
            self.assertFalse((mixed / "config_snapshot/files").exists())
            if uri:
                from mlflow import MlflowClient

                client = MlflowClient(tracking_uri=uri)
                ids = json.loads((mixed / "tracking.json").read_text())
                for name, status in [("unavailable", "FAILED"), ("available", "FINISHED")]:
                    self.assertEqual(client.get_run(ids[name]).info.status, status)
                    self.assertFalse(client.list_artifacts(ids[name], "recipe"))
                run_id = json.loads((first / "tracking.json").read_text())["ridge"]
                exp = client.get_experiment_by_name("photoz")
                assert exp is not None
                (old_run,) = client.search_runs([exp.experiment_id])
                for identifier, bundle in [(run_id, archive), (old_run.info.run_id, legacy)]:
                    remote = Path(
                        client.download_artifacts(identifier, "config_snapshot", str(root))
                    )
                    self.assertEqual(
                        (remote / "manifest.json").read_bytes(),
                        (bundle / "manifest.json").read_bytes(),
                    )
                    manifest = json.loads((bundle / "manifest.json").read_text())["sha256"]
                    for key in manifest:
                        self.assertEqual((remote / key).read_bytes(), (bundle / key).read_bytes())
