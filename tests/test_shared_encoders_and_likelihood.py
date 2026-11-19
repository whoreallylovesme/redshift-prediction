import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from lib.dataloaders.batches import ShuffledRowBatches
from lib.models.contrastive import ProjectionHead
from lib.models.encoders import EncodedModality, MaskedTableEncoder
from lib.train.likelihood import LikelihoodConfig, LikelihoodModule, row_loader


class SensorLikelihood(LikelihoodModule):
    def nll(self, batch: list[Tensor], *, training: bool) -> Tensor:
        estimates: Tensor = self.model(batch[0]).squeeze(1)
        return F.gaussian_nll_loss(estimates, batch[1], torch.ones_like(batch[1]))


class SharedEncoderAndLikelihoodTests(TestCase):
    def test_masked_sensor_channels_and_shared_or_per_row_descriptors(self) -> None:
        table = MaskedTableEncoder(4, 3, 5, 7, width=8, heads=2, layers=1)
        projection = ProjectionHead(7, 3)
        measurements = torch.randn(3, 4, 3)
        mask = torch.tensor([[False] * 4, [True, False, True, True], [True] * 4])
        descriptors = torch.randn(4, 5)
        clean = table(measurements, mask, descriptors)
        measurements[~mask] = torch.nan
        missing = table(measurements, mask, descriptors.expand(3, -1, -1))
        self.assertEqual(clean.features.shape, (3, 7))
        torch.testing.assert_close(missing.features, clean.features)
        torch.testing.assert_close(missing.available, mask.any(1))
        embedding: Tensor = projection(missing)
        torch.testing.assert_close(embedding[0], torch.zeros(3))
        torch.testing.assert_close(embedding[1:].norm(dim=1), torch.ones(2))
        embedding[:, 0].sum().backward()
        self.assertIsNotNone(table.token[0].weight.grad)

    def test_integer_sensor_values_keep_feature_dtype_when_all_rows_are_missing(self) -> None:
        encoder = MaskedTableEncoder(3, 2, 1, 4, width=4, layers=1)
        readings = torch.ones(2, 3, 2, dtype=torch.int64)
        descriptors = torch.ones(3, 1)
        mask = torch.ones(2, 3, dtype=torch.bool)
        measured = encoder(readings, mask, descriptors)
        missing = encoder(readings, ~mask, descriptors)
        torch.testing.assert_close(missing.features, torch.zeros_like(measured.features))

    def test_two_loader_workers_return_each_row_once_per_epoch(self) -> None:
        rows = np.arange(2**60, 2**60 + 11, dtype=np.int64)
        batches = ShuffledRowBatches(rows, seed=11, size=3, drop_last=False)
        loader = DataLoader(
            batches, num_workers=2, batch_size=None, collate_fn=np.asarray,
            persistent_workers=True, generator=torch.Generator().manual_seed(194),
        )
        first = list(loader)
        self.assertEqual([len(batch) for batch in first], [3, 3, 3, 2])
        first_epoch = np.concatenate(first)
        second_epoch = np.concatenate(list(loader))
        for epoch in (first_epoch, second_epoch):
            np.testing.assert_array_equal(np.sort(epoch), rows, strict=True)
        self.assertFalse(np.array_equal(first_epoch, second_epoch))

    def test_projection_ignores_unavailable_features_before_backpropagation(self) -> None:
        head = ProjectionHead(3, 2)
        features = torch.tensor([[1.0, 2.0, 3.0], [torch.nan] * 3], requires_grad=True)
        mask = torch.tensor([True, False])
        result = head(EncodedModality(features, mask))
        torch.testing.assert_close(result[1], torch.zeros(2))
        result[0, 0].backward()
        for parameter in head.parameters():
            assert parameter.grad is not None
            self.assertTrue(torch.isfinite(parameter.grad).all())
        assert features.grad is not None
        torch.testing.assert_close(features.grad[1], torch.zeros(3))
        with self.assertRaisesRegex(ValueError, "boolean"):
            head(EncodedModality(features, mask[:, None]))

    def test_gaussian_regression_can_fit_without_a_projection_file(self) -> None:
        cfg = LikelihoodConfig(epochs=1, batch_size=3)
        x = torch.arange(-3, 3, dtype=torch.float32)
        dataset = TensorDataset(x[:, None], x / 2)
        train = row_loader(dataset, np.arange(4), cfg, shuffle=True)
        held = row_loader(dataset, np.arange(4, 6), cfg)
        module = SensorLikelihood(nn.Linear(1, 1), cfg)
        with TemporaryDirectory() as directory:
            out = Path(directory) / "fit"
            module.fit_selected(out, train, held, {"target": "temperature"})
            report = json.loads((out / "train_report.json").read_text())
            self.assertEqual(report["selected_epoch"], 1)
            self.assertAlmostEqual(
                report["validation_nll"], float(module.nll(next(iter(held)), training=False).item())
            )
            state = torch.load(out / "selected_state.pt", weights_only=True)
            torch.testing.assert_close(state, module.model.state_dict())
