import subprocess
import sys
from unittest import TestCase

import numpy as np
import pandas as pd

from lib.report.evaluation import interval_for_delta, paired
from lib.report.task import REGRESSION, Floats, Metric, ReportTask
from photoz.bootstrap import arm_boot, seed_tile_boot, tile_boot
from photoz.data.euclid_zero_points import tile_draws


class BootstrapCompatibilityTests(TestCase):
    def setUp(self) -> None:
        self.tiles = np.array([7, 3, 7, 2, 2, 2, 2])
        self.z = np.arange(7) / 4
        errors = (
            np.array(
                [
                    [0, 1, -2, 3, -4, 2, 6],
                    [-2, 0, 3, 6, -1, -3, 2],
                    [5, 4, 0, -4, 2, 1, -2],
                ]
            )
            / 8
        )
        self.pred = self.z + (1 + self.z) * errors

    def test_public_bootstrap_works_without_the_astronomy_package(self) -> None:
        subprocess.run(
            [sys.executable, "-c", """
import sys
sys.modules["photoz"] = None
from lib.metrics import *
from lib.metrics.bootstrap import *
draws = bootstrap_distribution(lambda indices: indices.mean(), (4,), 5, 123)
assert draws.shape == (5,)
"""], check=True,
        )

    def test_paired_tile_intervals_and_calibration_counts(self) -> None:
        a, b, _ = self.pred
        np.testing.assert_array_equal(
            tile_boot(a, b, self.z, self.tiles, 17, 12),
            [-0.24092249999999996, 0.09266250000000004],
        )
        np.testing.assert_array_equal(
            tile_boot(a, None, self.z, self.tiles, 17, 12),
            [0.185325, 0.555975],
        )
        labels, inverse, counts = tile_draws(self.tiles, nboot=5, seed=12)
        np.testing.assert_array_equal(labels, [2, 3, 7])
        np.testing.assert_array_equal(inverse, [2, 1, 2, 0, 0, 0, 0])
        np.testing.assert_array_equal(
            counts, [[1, 1, 1], [2, 0, 1], [2, 1, 0], [1, 2, 0], [0, 1, 2]]
        )

    def test_seeds_and_tiles_keep_their_original_sampling_order(self) -> None:
        a, b, _ = self.pred
        one_tile = np.ones_like(self.tiles)
        cases = [
            (
                self.pred,
                self.pred[[2, 0]],
                self.tiles,
                [[-0.32123, 0.40771499999999994], [-0.40714285714285714, 0.39]],
            ),
            (
                self.pred,
                b,
                self.tiles,
                [[-0.38918249999999993, 0.06177499999999997], [-0.85, 0.23888888888888887]],
            ),
            (
                self.pred,
                b,
                one_tile,
                [[-0.2471, 0.08648499999999991], [-0.4285714285714286, 0.1428571428571428]],
            ),
            (a[None], self.z, one_tile, [[0.37065, 0.37065], [5 / 7, 5 / 7]]),
        ]
        for left, right, tiles, expected in cases:
            with self.subTest(left=left.shape, right=right.shape, tiles=len(np.unique(tiles))):
                np.testing.assert_array_equal(
                    seed_tile_boot(left, right, self.z, tiles, 17, 12), expected
                )
        np.testing.assert_array_equal(
            tile_boot(a, self.z, self.z, one_tile, 17, 12), [0.37065, 0.37065]
        )

    def test_arms_with_equal_seed_counts_share_draws(self) -> None:
        models = {"three": self.pred, "two": self.pred[:2], "twin": self.pred[:2].copy()}
        refs = {"first": self.pred[1], "second": self.pred[0]}
        draws = arm_boot(models, refs, self.z, self.tiles, n_boot=5, seed=12)
        for ref in refs:
            np.testing.assert_array_equal(draws["two"][ref], draws["twin"][ref])
        np.testing.assert_array_equal(
            draws["two"]["first"],
            [
                [-0.09266249999999998, -0.28571428571428575],
                [0, 0],
                [0, 0],
                [0, 0],
                [-0.37065, -0.8],
            ],
        )
        np.testing.assert_array_equal(
            draws["three"]["second"],
            [
                [0, 0],
                [0, 1],
                [-0.18532500000000002, 0.09999999999999998],
                [-0.061775000000000024, 0.1428571428571428],
                [-0.12355, -0.4],
            ],
        )

    def test_report_omits_undefined_draws_but_propagates_numeric_nan(self) -> None:
        def some(truth: Floats, pred: Floats) -> float | None:
            return float(np.mean(pred)) if 0.25 in truth else None

        frame = pd.DataFrame(
            {
                "_truth": self.z,
                "_pred": self.pred[0],
                "_other": self.pred[1],
                "_group": self.tiles,
            }
        )
        task = ReportTask(
            "fixture",
            {
                "mean": Metric("mean", "min", lambda t, p: float(np.mean(p))),
                "some": Metric("some", "min", some),
                "nan": Metric("nan", "min", lambda t, p: some(t, p) if 0.25 in t else np.nan),
                "never": Metric("never", "min", lambda t, p: None),
            },
        )
        interval = interval_for_delta(frame, task, 17)
        assert interval is not None
        self.assertEqual(set(interval), {"mean", "some", "nan"})
        np.testing.assert_array_equal(interval["mean"], [-0.18375, 0.3125])
        np.testing.assert_array_equal(interval["some"], [-0.20625, 0.2864583333333335])
        self.assertTrue(np.isnan(interval["nan"]).all())
        self.assertIsNone(interval_for_delta(frame.assign(_group=1), task, 17))

    def test_paired_report_can_read_object_groups_from_either_method(self) -> None:
        a = pd.DataFrame({"_truth": self.z, "_pred": self.pred[0], "_evaluated": True})
        b = a.assign(_pred=self.pred[1], _group=self.tiles).iloc[::-1]
        grouped = paired(a.assign(_group=self.tiles), b, REGRESSION, {}, {}, 17)["ci95"]
        self.assertIsNotNone(grouped)
        self.assertEqual(paired(a, b, REGRESSION, {}, {}, 17)["ci95"], grouped)
        reverse = paired(b, a, REGRESSION, {}, {}, 17)["ci95"]
        for metric, interval in grouped.items():
            np.testing.assert_allclose(reverse[metric], -np.array(interval)[::-1])
