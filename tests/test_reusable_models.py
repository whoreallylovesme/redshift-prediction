import pickle
from unittest import TestCase

import numpy as np
import torch

from lib.models.density import BinnedDensityHead, piecewise_uniform_nll, posterior_quantiles
from lib.models.encoders import ResidualBlock, ResidualMLP, attention_stack, regression_head
from lib.models.graph import ResidualMessage, weighted_neighbors
from lib.models.masked import MaskedPrediction, masked_gaussian_nll, sample_mask


class ReusableModelTests(TestCase):
    def test_signed_bounded_target_and_historical_redshift_limits(self) -> None:
        from lib.models.targets import bounded_logit, scaled_sigmoid
        from photoz.targets import logistic_to_z, z_to_logistic

        signed = torch.tensor([-4.5, 0.0, 7.0], dtype=torch.float64, requires_grad=True)
        restored = scaled_sigmoid(bounded_logit(signed, -5, 10), -5, 10)
        torch.testing.assert_close(restored, signed)
        restored.sum().backward()
        torch.testing.assert_close(signed.grad, torch.ones_like(signed))

        z = torch.tensor([-0.02, 0.0, 0.2, 0.4, 0.42])
        historical = torch.log(z.clamp_min(1e-6) / (0.4 - z).clamp_min(1e-6))
        torch.testing.assert_close(z_to_logistic(z), historical, rtol=0, atol=0)
        torch.testing.assert_close(logistic_to_z(historical), 0.4 * historical.sigmoid())

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

    def test_gaussian_nll_preserves_tiny_variances_and_equal_weight_per_active_row(self) -> None:
        hidden = torch.tensor([[True, True, False], [False, True, False], [False, False, False]])
        valid = torch.tensor([[True, True, True], [True, True, False], [False, False, False]])
        mean = torch.full((3, 3), torch.nan, dtype=torch.float64)
        variance, target, error = (mean.clone() for _ in range(3))
        mean[hidden] = torch.tensor([0.0, 2.0, 4.0], dtype=mean.dtype)
        variance[hidden] = torch.tensor([1e-14, 1.0, 4.0], dtype=mean.dtype)
        target[hidden] = torch.tensor([1e-7, 0.0, 8.0], dtype=mean.dtype)
        error[hidden] = torch.tensor([1e-7, 0.5, 2.0], dtype=mean.dtype)
        mean.requires_grad_()
        variance.requires_grad_()
        prediction = MaskedPrediction(mean, variance, torch.empty(3, 2))
        actual = masked_gaussian_nll(prediction, target, error, valid, hidden)
        sigma = (variance[hidden] + error[hidden].square()).sqrt()
        dist = torch.distributions.Normal(mean[hidden], sigma)
        nll = -dist.log_prob(target[hidden])
        expected = (nll[:2].mean() + nll[2]) / 2
        torch.testing.assert_close(actual, expected)
        actual_grads = torch.autograd.grad(actual, (mean, variance))
        expected_grads = torch.autograd.grad(expected, (mean, variance))
        for grad, reference in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(grad, reference)
            self.assertTrue((grad[~hidden] == 0).all())
