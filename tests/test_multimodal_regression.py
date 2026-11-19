from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lib.config import PytorchTrainConfig
from lib.inference.images import predict_catalog
from lib.train.regression_lightning import fit_images


class MultimodalRegressionTests(TestCase):
    def test_fusion_fits_training_statistics_and_preserves_held_ids(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = Path("tests/fixtures/fusion")
            cfg = PytorchTrainConfig.model_validate({
                "model_type": "lightning", "device": "cpu", "estimator": "fusion_cnn",
                "estimator_params": {"image_channels": 1, "features": 2, "width": 8},
                "data_dir": str(data / "images"), "catalog": str(data / "train.csv"),
                "image_catalog": {
                    "id_column": "sample", "target_column": "response",
                    "features": ["time", "density"],
                },
                "out_size": 4, "epochs": 2, "batch_size": 3, "num_workers": 0,
                "lr": 0.001, "train_ratio": 0.75, "seed": 194,
            })
            training = pd.read_csv(cfg.catalog)
            fitted, _ = train_test_split(
                training, train_size=cfg.train_ratio, random_state=cfg.seed
            )
            weights = fit_images(cfg, root / "fit")
            with np.load(root / "fit/image_normalization.npz", allow_pickle=False) as norms:
                self.assertEqual(norms["features"].tolist(), ["time", "density"])
                np.testing.assert_allclose(
                    norms["feature_mean"], fitted[["time", "density"]].mean()
                )
            held = data / "held.csv"
            prediction = root / "held.parquet"
            predict_catalog(
                cfg, cfg.image_catalog, weights, data / "images", held, prediction, "sample"
            )
            expected = pd.read_csv(held)
            actual = pd.read_parquet(prediction)
            self.assertEqual(expected["sample"].tolist(), actual["sample"].tolist())
            np.testing.assert_allclose(expected.response, actual.y_true)
            self.assertTrue(np.isfinite(actual.y_pred).all())
            cfg.image_catalog.features.reverse()
            with self.assertRaisesRegex(ValueError, "feature columns differ"):
                predict_catalog(
                    cfg, cfg.image_catalog, weights, data / "images", held,
                    root / "invalid.parquet", "sample",
                )
