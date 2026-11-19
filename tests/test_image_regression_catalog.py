from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from lib.config.schemas import ImageCatalogConfig
from lib.dataloaders.images import catalog
from lib.inference import inference
from lib.models.factories import FACTORIES
from lib.report import build_report_data
from runtime.train import train


def tiny_image_model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(16, 1))


class ImageRegressionTests(TestCase):
    def test_generic_training_and_prediction(self) -> None:
        with TemporaryDirectory() as temp, patch.dict(FACTORIES, tiny_fixture=tiny_image_model):
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
            yaml_path = root / "image.yaml"
            output = root / "model"
            yaml_path.write_text(
                yaml.safe_dump(
                    {
                        "model_type": "lightning",
                        "device": "cpu",
                        "estimator": "tiny_fixture",
                        "data_dir": str(training),
                        "catalog": str(training_rows),
                        "out_size": 4,
                        "image_catalog": {
                            "id_column": "sample",
                            "target_column": "mass",
                            "id_dtype": "string",
                            "array_key": "data",
                        },
                        "train": {
                            "epochs": 1,
                            "batch_size": 2,
                            "lr": 0.01,
                            "weight_decay": 0.0,
                            "freeze_epochs": 0,
                            "num_workers": 0,
                            "train_ratio": 0.75,
                            "seed": 4,
                        },
                    }
                )
            )
            train(str(yaml_path), outdir=str(output))
            self.assertTrue((output / "best_model.pt").is_file())
            with np.load(output / "image_normalization.npz") as stats:
                from sklearn.model_selection import train_test_split

                indices, _ = train_test_split(range(len(ids)), train_size=0.75, random_state=4)
                self.assertAlmostEqual(float(stats["mean"][0]), float(np.mean(indices)), places=5)
            for name in ("099", "088"):
                np.savez(testing / f"{name}.npz", data=np.full((5, 6), 20, dtype=np.float32))
            by_id = root / "predict.parquet"
            pd.DataFrame({"sample": ["099", "088"]}).to_parquet(by_id, index=False)
            pred = root / "predictions.parquet"
            inference(
                str(output / "best_model.pt"),
                str(pred),
                str(yaml_path),
                data_dir=str(testing),
                catalog=str(by_id),
                held_out=True,
            )
            result = pd.read_parquet(pred)
            self.assertEqual(list(result.columns), ["sample", "y_pred"])
            self.assertEqual(list(result["sample"]), ["099", "088"])
            self.assertTrue(np.isfinite(result.y_pred).all())
            descriptor = {"name": "tiny", "split": "test", "predictions": pred}
            report = build_report_data([descriptor], "regression", "Images")
            method = report["rows"][0]
            self.assertTrue(method["paired_id"])
            self.assertEqual(
                method["details"]["image_split_overlap"], {"train": 0, "validation": 0}
            )
            with self.assertRaisesRegex(ValueError, "overlap model development IDs"):
                inference(
                    str(output / "best_model.pt"),
                    str(root / "contaminated.parquet"),
                    str(yaml_path),
                    catalog=str(training_rows),
                    held_out=True,
                )
            self.assertFalse((root / "contaminated.parquet").exists())

    def test_catalog_rejects_duplicate_ids(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "duplicates.parquet"
            pd.DataFrame({"id": [2**60, 2**60]}).to_parquet(path, index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                catalog(path, ImageCatalogConfig(id_column="id", target_column="truth"))
