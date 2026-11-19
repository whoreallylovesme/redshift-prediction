import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from lib.config.schemas import ImageCatalogConfig, PytorchTrainConfig
from lib.dataloaders.images import catalog
from lib.inference.images import predict_catalog
from lib.models.registry import NN_FACTORIES
from lib.utils.utils import load_yaml_as
from runtime.pipeline import pipeline


def tiny_image_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 1))


class ImageRegressionTests(TestCase):
    def test_generic_training_and_prediction(self) -> None:
        with TemporaryDirectory() as temp, patch.dict(NN_FACTORIES, tiny_fixture=tiny_image_model):
            root = Path(temp)
            training, testing = root / "training", root / "testing"
            training.mkdir()
            testing.mkdir()
            ids = [f"{n:03}" for n in range(8)]
            mass = [float(n - 4) for n in range(8)]
            for index, stem in enumerate(ids):
                np.savez(training / f"{stem}.npz", data=np.full((4, 4), index, dtype=np.float32))
            training_rows = root / "train.csv"
            pd.DataFrame({"sample": ids, "mass": mass}).to_csv(training_rows, index=False)
            for name in ("099", "088"):
                np.savez(testing / f"{name}.npz", data=np.full((5, 6), 20, dtype=np.float32))
            by_id = root / "predict.parquet"
            pd.DataFrame({"sample": ["099", "088"]}).to_parquet(by_id, index=False)
            experiment = root / "experiment.yaml"
            experiment.write_text(yaml.safe_dump({
                "name": "sensor images",
                "output": str(root / "runs"),
                "cache": False,
                "dataset": {
                    "source": str(training_rows), "evaluation_files": {"held": str(by_id)},
                    "id_column": "sample", "target": "mass", "array_key": "data",
                    "images": {"train": str(training), "held": str(testing)}, "image_size": 4,
                },
                "methods": {
                    "tiny": {
                        "kind": "lightning", "model": "tiny_fixture", "device": "cpu",
                        "train": {
                            "epochs": 1, "batch_size": 2, "lr": 0.01, "num_workers": 0,
                            "train_ratio": 0.75, "seed": 4,
                        },
                        "report": {"truth_column": None},
                    }
                },
            }))
            run = pipeline(str(experiment))
            output = run / "methods/tiny"
            weights = output / "best_model.pt"
            self.assertTrue(weights.is_file())
            with np.load(output / "image_normalization.npz") as stats:
                from sklearn.model_selection import train_test_split

                indices, _ = train_test_split(range(len(ids)), train_size=0.75, random_state=4)
                self.assertAlmostEqual(float(stats["mean"][0]), float(np.mean(indices)), places=5)
            pred = output / "predictions_held.parquet"
            result = pd.read_parquet(pred)
            self.assertEqual(list(result.columns), ["sample", "y_pred"])
            self.assertEqual(list(result["sample"]), ["099", "088"])
            self.assertTrue(np.isfinite(result.y_pred).all())
            report = json.loads((run / "report-data.json").read_text())
            method = report["rows"][0]
            self.assertTrue(method["paired_id"])
            self.assertEqual(
                method["details"]["image_split_overlap"], {"train": 0, "validation": 0}
            )
            recipe = run / "config_snapshot/resolved/tiny.yaml"
            cfg = load_yaml_as(recipe, PytorchTrainConfig)
            with self.assertRaisesRegex(ValueError, "overlap model development IDs"):
                predict_catalog(
                    cfg, cfg.image_catalog, weights, training, training_rows,
                    root / "contaminated.parquet", None, held_out=True,
                )
            self.assertFalse((root / "contaminated.parquet").exists())

    def test_catalog_rejects_duplicate_ids(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "duplicates.parquet"
            pd.DataFrame({"id": [2**60, 2**60]}).to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                catalog(path, ImageCatalogConfig(id_column="id", target_column="truth"))
