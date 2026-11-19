from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd
import yaml
from prefect.testing.utilities import prefect_test_harness

from lib.config import Experiment, load_flow
from runtime.pipeline import pipeline


class SharedCatalogTest(TestCase):
    def test_image_and_table_fit_use_the_same_catalog_and_held_ids(self) -> None:
        # Use the public two-method template; 12 local fixtures keep this CPU regression test small.
        recipe = yaml.safe_load(Path("examples/configs/teddy.yaml").read_text())
        fixture = Path("tests/fixtures/fusion").resolve()
        with TemporaryDirectory() as tmp, prefect_test_harness():
            root = Path(tmp)
            catalog = root / "features.yaml"
            catalog.write_text("target: response\nfeature_sets: {sensors: [time, density]}\n")
            recipe.update(task="regression", output=str(root / "runs"), cache=False)
            recipe["dataset"] = {
                "source": str(fixture / "train.csv"),
                "evaluation_files": {"held": str(fixture / "held.csv")},
                "id_column": "sample",
                "feature_catalog": str(catalog),
                "feature_set": "sensors",
                "images": str(fixture / "images"),
                "image_size": 16,
            }
            recipe["methods"]["catboost"]["params"]["iterations"] = 3
            network = recipe["methods"]["capsnet"]
            network.update(
                device="cpu", train={"epochs": 1, "batch_size": 3, "lr": 0.0005, "num_workers": 0}
            )
            network["params"].update(image_channels=1, input_size=16, routings=1, dim_capsule=8)
            flow = root / "study.yaml"
            flow.write_text(yaml.safe_dump(recipe))
            load_flow(flow, check=True)
            run = pipeline(str(flow))
            assert run is not None
            snapshot = run / "config_snapshot"
            backend = yaml.safe_load((snapshot / "resolved/capsnet.yaml").read_text())
            self.assertEqual(Path(backend["catalog"]), run / "data/train.parquet")
            held = pd.read_parquet(run / "data/held_held.parquet").set_index("sample")
            self.assertEqual(len(held), 4)
            for model in ("catboost", "capsnet"):
                preds = pd.read_parquet(run / "methods" / model / "predictions_held.parquet")
                self.assertEqual(preds["sample"].tolist(), held.index.tolist())
            report = json.loads((run / "metrics.json").read_text())
            self.assertEqual([r["status"] for r in report["rows"]], ["completed"] * 2)
            replay = load_flow(snapshot / "flow.yaml", check=True)
            self.assertIsInstance(replay, Experiment)
            self.assertEqual(replay, load_flow(flow, check=True))
