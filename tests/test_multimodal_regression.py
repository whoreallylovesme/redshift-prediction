from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from lib.inference import inference
from runtime.train import train


class MultimodalRegressionTests(TestCase):
    def test_fusion_fits_training_statistics_and_preserves_held_ids(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            recipe = Path("configs/nn/fusion_example.yaml")
            cfg = yaml.safe_load(recipe.read_text())
            training = pd.read_csv(cfg["catalog"])
            fitted, _ = train_test_split(training, train_size=0.75, random_state=194)
            train(str(recipe), outdir=str(root / "fit"))
            with np.load(root / "fit/image_normalization.npz", allow_pickle=False) as norms:
                self.assertEqual(norms["features"].tolist(), ["time", "density"])
                np.testing.assert_allclose(
                    norms["feature_mean"], fitted[["time", "density"]].mean()
                )
            held = Path("tests/fixtures/fusion/held.csv")
            prediction = root / "held.parquet"
            weights = str(root / "fit/best_model.pt")
            inference(
                weights,
                str(prediction),
                str(recipe),
                data_dir=cfg["data_dir"],
                catalog=str(held),
                id_column="sample",
            )
            expected = pd.read_csv(held)
            actual = pd.read_parquet(prediction)
            self.assertEqual(expected["sample"].tolist(), actual["sample"].tolist())
            np.testing.assert_allclose(expected.response, actual.y_true)
            self.assertTrue(np.isfinite(actual.y_pred).all())
            cfg["image_catalog"]["features"].reverse()
            changed = root / "changed.yaml"
            changed.write_text(yaml.safe_dump(cfg))
            with self.assertRaisesRegex(ValueError, "feature columns differ"):
                inference(
                    weights,
                    str(root / "invalid.parquet"),
                    str(changed),
                    data_dir=cfg["data_dir"],
                    catalog=str(held),
                    id_column="sample",
                )
