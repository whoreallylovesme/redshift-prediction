from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from lib.config import PytorchTrainConfig
from photoz.methods.capsnet import _predict
from photoz.methods.teddy_images import predict_teddy
from photoz.models.capsnet import CapsNet
from photoz.models.predictors import CapsNetPredictor


class Average(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.mean(dim=(1, 2, 3))


class LegacyPredictTest(TestCase):
    def test_teddy_uses_saved_training_scale_and_accounts_for_bad_images(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images"
            images.mkdir()
            np.savez(images / "0001.npz", image=np.full((8, 8, 4), 6, dtype="float32"))
            (images / "0002.npz").write_bytes(b"bad image")
            np.savez(root / "image_normalization.npz", mean=np.full(4, 2), std=np.full(4, 2))
            weights = root / "best_model.pt"
            torch.save(Average().state_dict(), weights)
            cfg = PytorchTrainConfig.model_validate(
                yaml.safe_load(Path("tests/fixtures/legacy/models/fusion.yaml").read_text())
            )
            cfg.device = "cpu"
            out = root / "held.parquet"
            with patch("photoz.methods.teddy_images.load_estimator", return_value=Average()):
                predict_teddy(str(weights), cfg, str(images), None, str(out), "oid")
            prediction = pd.read_parquet(out)
            self.assertEqual(prediction.oid.tolist(), ["0001", "0002"])
            self.assertAlmostEqual(prediction.y_pred.iloc[0], 2.0)
            self.assertTrue(np.isnan(prediction.y_pred.iloc[1]))
            self.assertNotEqual(prediction.status.iloc[1], "ok")

    def test_capsnet_predict_matches_single_batch_api(self) -> None:
        with TemporaryDirectory() as temp:
            weights = Path(temp) / "original.pt"
            raw = CapsNet(in_channels=4, input_size=64, num_class=2, dim_capsule=16, routings=3)
            torch.save(raw.state_dict(), weights)
            device = torch.device("cpu")
            predictor = CapsNetPredictor(str(weights), 4, 64, 0.0, 0.4, device)
            images = torch.rand(2, 4, 64, 64)
            expected = predictor.predict(images)
            dataset = TensorDataset(images, torch.zeros(2), torch.zeros(2))
            actual = _predict(predictor, DataLoader(dataset, batch_size=2))
            for name in expected:
                np.testing.assert_allclose(actual[name], expected[name].cpu().numpy(), atol=1e-6)
