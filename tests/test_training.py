import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lib.config import PytorchTrainConfig
from lib.train.regression import RegressionTask
from lib.train.regression_lightning import run_regression
from photoz.models.capsnet import CapsNet
from photoz.training import euclid_classical, labels
from photoz.training.clip_epochs import ShuffledRowBatches
from photoz.training.regression import photoz_regression_task


class TrainingTests(TestCase):
    def test_epoch_batches_keep_seed_and_tail_policy(self) -> None:
        for drop_last in (False, True):
            rows = np.arange(21, 34, dtype=np.int64)
            rng = np.random.default_rng(19)
            loader = ShuffledRowBatches(rows, 19, 4, drop_last).loader()
            for _ in range(2):
                expected = rng.permutation(rows)
                actual = np.concatenate([batch.numpy() for batch in loader])
                np.testing.assert_array_equal(actual, expected[:12] if drop_last else expected)

    def test_generic_and_photoz_checkpoint_selection(self) -> None:
        data = torch.arange(-4, 4, dtype=torch.float32)[:, None]
        loader = DataLoader(TensorDataset(data, data[:, 0] * 2), batch_size=4)
        cfg = PytorchTrainConfig.model_construct(
            epochs=2, freeze_epochs=0, weight_decay=0.01, lr=0.01, seed=8, head_prefix="0",
        )
        task = RegressionTask()
        self.assertEqual(photoz_regression_task(cfg).monitor, "val/nmad")
        with TemporaryDirectory() as directory:
            model = nn.Sequential(nn.Linear(1, 1))
            best = run_regression(
                cfg, model, loader, loader, directory, task=task, device=torch.device("cpu"),
            )
            for path in (best, Path(directory) / "last_model.pt"):
                state = torch.load(path, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
            history = json.loads((Path(directory) / "history.json").read_text())
            self.assertEqual([row["epoch"] for row in history], [1, 2])
            for row in history:
                for name in ("lr", "train/loss", "val/loss", "train/rmse", "val/rmse"):
                    self.assertTrue(np.isfinite(row[name]), name)
            ckpt = torch.load(Path(directory) / "weights/best.ckpt", weights_only=True)
            selections = [v for v in ckpt["callbacks"].values() if v.get("monitor")]
            self.assertEqual([v["monitor"] for v in selections], ["val/rmse"])

    def test_morphology_predictions_keep_order_in_partial_last_batch(self) -> None:
        class TinyProbabilities(nn.Module):
            def forward(
                self, images: torch.Tensor, target_class: torch.Tensor | None = None,
            ) -> tuple[None, torch.Tensor, None]:
                probability = images.flatten(1)[:, 0]
                return None, torch.stack((1 - probability, probability), dim=1), None

        values = torch.tensor([0.6, 0.1, 0.9])
        model = cast(CapsNet, TinyProbabilities())
        ds = TensorDataset(values[:, None], values, values)
        loader = DataLoader(ds, batch_size=2)
        probabilities = labels._predict_probs(model, loader, torch.device("cpu"))
        np.testing.assert_array_equal(probabilities, values.numpy())

    def test_annz_committee_matches_cpu_reference(self) -> None:
        rng = np.random.default_rng(49)
        x = rng.normal(size=(26, 4))
        y = 0.3 * x[:, 1] + 0.2 * x[:, 2]
        q = rng.normal(size=(4, 4))
        with patch.object(euclid_classical, "ANNZ_COMMITTEE", 2), patch.object(
            euclid_classical, "ANNZ_STEPS", 3
        ):
            predictions = euclid_classical.fit_annz(x, y, np.arange(len(y)), q, 1)
        expected = [-0.08305140662, 0.10009164482, -0.01424824797, -0.11387139924]
        np.testing.assert_allclose(predictions, expected, rtol=1e-5, atol=1e-7)
