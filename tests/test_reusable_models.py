import pickle
from unittest import TestCase

import numpy as np
import torch

from lib.models.density import BinnedDensityHead, piecewise_uniform_nll, posterior_quantiles
from lib.models.encoders import ResidualBlock, ResidualMLP, attention_stack, regression_head
from lib.models.graph import ResidualMessage, weighted_neighbors
from lib.models.masked import MaskedPrediction, masked_gaussian_nll, sample_mask


class ReusableModelTests(TestCase):
    def test_old_serialized_class_names_resolve_to_current_estimators(self) -> None:
        from lib.models.gam import GAMRegressor
        from lib.models.llr import LocalLinearRegression

        self.assertIs(pickle.loads(b"clib.gam\nGAMRegressor\n."), GAMRegressor)
        self.assertIs(pickle.loads(b"clib.llr\nLocalLinearRegression\n."), LocalLinearRegression)

    def test_residual_attention_and_graph_on_arbitrary_feature_dimensions(self) -> None:
        values = torch.randn(3, 7)
        mlp = ResidualMLP(7, 6, width=10, depth=3)
        nodes = mlp(values)
        neighbors = torch.tensor([[1, 2], [0, 2], [0, 1]])
        weights = torch.tensor([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        context = weighted_neighbors(nodes, neighbors, weights)
        torch.testing.assert_close(context[0], nodes[2])
        messages = ResidualMessage(width=6)(nodes, context)
        tokens = attention_stack(1, width=6, heads=2)(messages[None])
        out = regression_head(6, hidden=(9, 5), outputs=2)(tokens)
        self.assertEqual(out.shape, (1, 3, 2))
        out.square().mean().backward()
        self.assertIsNotNone(mlp.input.weight.grad)
        images = torch.randn(3, 2, 9, 7)
        self.assertEqual(ResidualBlock(2, 5, 2, groups=1)(images).shape, (3, 5, 5, 4))
        with self.assertRaisesRegex(ValueError, "positive"):
            weighted_neighbors(nodes, neighbors, torch.zeros_like(weights))

    def test_signed_scalar_target_density_and_masked_sensor_likelihood(self) -> None:
        edges = np.array([-22.0, -4.0, 3.0, 15.0])
        head = BinnedDensityHead(2, edges, hidden_dim=8)
        prediction = head(torch.randn(3, 2))
        loss = piecewise_uniform_nll(prediction.logits, torch.tensor([-22.0, -3.0, 15.0]), edges)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        median = posterior_quantiles(prediction.bin_mass.detach().numpy(), edges, [0.5])
        self.assertTrue(np.all((median >= -22.0) & (median <= 15.0)))
        valid = torch.tensor([[True, False, True], [True, True, True]])
        hidden = sample_mask(valid, generator=torch.Generator().manual_seed(8))
        means = torch.zeros((2, 3), requires_grad=True)
        predicted = MaskedPrediction(means, torch.ones_like(means), torch.zeros((2, 4)))
        nll = masked_gaussian_nll(
            predicted, torch.ones_like(means), torch.full_like(means, 0.2), valid, hidden
        )
        nll.backward()
        self.assertIsNotNone(means.grad)
        assert means.grad is not None
        self.assertTrue((means.grad[~hidden] == 0).all())
