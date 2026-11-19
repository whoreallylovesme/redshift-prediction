from __future__ import annotations

import csv
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lib.report import REGRESSION, build_report_data, render_report
from photoz.metrics import evaluate_regression
from photoz.report import PHOTOZ


class ReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.path = Path(self.work.name)
        self.ids = [2**60 + i for i in range(5)]
        self.ref = pd.DataFrame(
            {
                "id": self.ids,
                "truth": [0.0, 1.0, 2.0, 2.0, 1.0],
                "group": ["a", "a", "b", "b", "c"],
                "family": ["X", "Y", "X", "Y", "X"],
                "tab": [1, 1, 1, 1, 0],
                "image": [0, 1, 0, 1, 1],
                "f": [1.0, 2.0, np.nan, 1.0, 2.0],
            }
        )
        self.ref.to_parquet(self.path / "ref.parquet")
        for name, ids, pred in [
            ("A", [2, 0, 1, 3], [2.0, 0.15, 1.0, 1.8]),
            ("B", [0, 3, 1, 4], [0.2, 2.0, 1.3, 1.0]),
        ]:
            pd.DataFrame(
                {
                    "id": [self.ids[i] for i in ids],
                    "z": pred,
                    "ok": ["ok"] * 3 + ["invalid"],
                    "lower": np.array(pred) - 0.2,
                    "upper": np.array(pred) + 0.2,
                }
            ).to_parquet(self.path / f"{name}.parquet")
        self.methods: list[dict[str, Any]] = [
            {
                "name": n,
                "split": "held",
                "role": "held-out",
                "predictions": self.path / f"{n}.parquet",
                "reference": self.path / "ref.parquet",
                "id_column": "id",
                "truth_column": None,
                "reference_truth_column": "truth",
                "prediction_column": "z",
                "group_column": "group",
                "status_column": "ok",
                "slices": ["family"],
                "modality_columns": {"T": "tab", "I": "image"},
                "feature_columns": ["f"],
                "lower_column": "lower",
                "upper_column": "upper",
            }
            for n in ("A", "B")
        ]

    def test_all_rows_paired_by_exact_id_and_slice(self) -> None:
        d = build_report_data(self.methods, PHOTOZ, "validation", {"bootstrap_replicates": 6})
        a = d["rows"][0]
        self.assertEqual(
            a["coverage"],
            {
                "requested": 5,
                "admitted": 4,
                "predicted": 3,
                "evaluated": 3,
                "reasons": {"invalid": 1, "missing prediction row": 1},
            },
        )
        self.assertEqual(a["data"]["modalities"], {"T": 2, "T + I": 2, "I": 1})
        self.assertEqual(a["data"]["missing_features"]["f"], 1)
        self.assertAlmostEqual(a["metrics"]["f_outlier_percent"], 100 / 3)
        r = d["pairs"][0]["slices"]["All objects"]
        self.assertEqual((r["shared_ids"], r["count"]), (5, 2))
        self.assertEqual({e["id"] for e in r["examples"]}, {str(k) for k in self.ids[:2]})
        self.assertAlmostEqual(r["b"]["mae"], 0.25)
        self.assertIsNone(r["ci95"])
        np.testing.assert_allclose(
            np.array(list(PHOTOZ.guides(np.array([1.0])).values())), np.array([[1.0], [1.3], [0.7]])
        )
        self.assertEqual(d["pairs"][0]["slices"]["family = X"]["count"], 1)
        self.assertEqual(a["slices"]["All objects"]["charts"]["interval"]["covered"], 3)

    def test_html_exports_roundtrip_all_metrics_and_states(self) -> None:
        history = [{"epoch": 1, "valid": {"loss": 0.7}}, {"epoch": 2, "valid": {"loss": 0.6}}]
        self.methods[0]["history"] = history
        methods = [
            *self.methods,
            {"name": "failed", "split": "held", "status": "failed", "reason": "failure"},
        ]
        d = render_report(self.path, methods, PHOTOZ, "A </script>")
        html = (self.path / "report.html").read_text()
        match = re.search(
            r'<script type="application/json" id="report-data">(.*?)</script>', html, re.S
        )
        self.assertIsNotNone(match)
        data = json.loads(match.group(1))  # type: ignore[union-attr]
        exported = json.loads((self.path / "metrics.json").read_text())
        self.assertEqual(data["title"], "A </script>")
        self.assertNotIn("<script src=", html)
        self.assertEqual(data["rows"][0]["history"][1]["valid.loss"], 0.6)
        self.assertIsNone(exported["rows"][2]["coverage"])
        self.assertEqual(d["rows"][0]["metrics"], exported["rows"][0]["metrics"])
        with (self.path / "metrics.csv").open() as stream:
            first = next(csv.DictReader(stream))
        for k, value in d["rows"][0]["metrics"].items():
            self.assertEqual(float(first[k]), value)
        self.assertEqual(d["metrics"]["r2"]["direction"], "max")
        self.assertEqual(d["metrics"]["bias"]["direction"], "zero")

    def test_legacy_and_unlabeled(self) -> None:
        m = {
            "name": "legacy",
            "split": "dev",
            "id_column": None,
            "predictions": pd.DataFrame({"y_true": [4.0, 3.0], "y_pred": [5.0, 4.0]}),
        }
        b = {
            "name": "no labels",
            "split": "dev",
            "truth_column": None,
            "predictions": pd.DataFrame({"object_id": [3], "y_pred": [4.0]}),
        }
        d = build_report_data([m, b], "regression", "counts")
        self.assertEqual(d["rows"][0]["metrics"]["mae"], 1.0)
        self.assertEqual(d["rows"][1]["coverage"]["predicted"], 1)
        self.assertEqual(d["rows"][1]["coverage"]["evaluated"], 0)
        self.assertFalse(d["pairs"][0]["slices"])
        self.assertEqual(set(d["rows"][0]["metrics"]), set(REGRESSION.metrics))

    def test_invalid_ids_and_truth_mismatch(self) -> None:
        for ids in ([1.0, 2.0], [1, 1], [1, None]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                build_report_data(
                    [
                        {
                            "name": "a",
                            "split": "x",
                            "predictions": pd.DataFrame(
                                {"object_id": ids, "y_pred": [0.1, 0.1], "y_true": [0.1, 0.1]}
                            ),
                        }
                    ],
                    "regression",
                    "bad",
                    strict=True,
                )
        for m in self.methods:
            m.pop("reference")
            m["truth_column"] = "truth"
        for i, m in enumerate(self.methods):
            m["predictions"] = pd.DataFrame(
                {
                    "id": [4],
                    "z": [1.0],
                    "truth": [1.0 + i],
                    "ok": ["ok"],
                    "family": ["x"],
                    "f": [0],
                    "tab": [1],
                    "image": [0],
                    "group": ["t"],
                    "lower": [0.5],
                    "upper": [1.5],
                }
            )
        with self.assertRaisesRegex(ValueError, "inconsistent reference"):
            build_report_data(self.methods, PHOTOZ, "bad", strict=True)

    def test_failure_isolated_and_comparison_rejected(self) -> None:
        missing = dict(self.methods[0], name="broken", predictions=self.path / "missing.parquet")
        d = build_report_data([missing, *self.methods], PHOTOZ, "compare")
        self.assertEqual(d["rows"][0]["status"], "failed")
        self.assertIn("FileNotFoundError", d["rows"][0]["details"]["reason"])
        self.assertEqual(d["rows"][1]["status"], "completed")
        base = {"split": "dev", "reference": None, "truth_column": "y_true"}
        f = pd.DataFrame({"object_id": [1], "y_true": [1.0], "y_pred": [0.9]})
        d = build_report_data(
            [
                dict(base, name="A", predictions=f),
                dict(base, name="B", predictions=f.assign(y_true=2.0)),
            ],
            PHOTOZ,
            "bad",
        )
        self.assertEqual(d["pairs"][0]["status"], "failed")
        self.assertIn("inconsistent", d["pairs"][0]["reason"])

    def test_integer_status_is_preserved_across_an_incomplete_reference_join(self) -> None:
        row = build_report_data([{
            "name": "m", "split": "s", "id_column": "id", "truth_column": "truth",
            "predictions": pd.DataFrame({"id": [1, 2], "y_pred": [0.5, 99.0], "status": [0, 9]}),
            "reference": pd.DataFrame({"id": [1, 2, 3], "truth": [0.5, 1.0, 1.0]}),
            "ok_statuses": [0],
        }], REGRESSION, "Report", strict=True)["rows"][0]
        self.assertEqual(row["coverage"]["evaluated"], 1)
        self.assertEqual(row["coverage"]["reasons"], {"9": 1, "missing prediction row": 1})
        self.assertEqual(row["metrics"]["mae"], 0)

    def test_metrics_match_historical_and_density_uses_all_rows(self) -> None:
        t = np.linspace(0.1, 3.0, 1021)
        p = t + np.linspace(-0.2, 0.6, len(t))
        m = {
            "name": "CV",
            "split": "held",
            "predictions": pd.DataFrame({"z_spec": t, "z_pred": p, "id": np.arange(len(t))}),
            "id_column": "id",
            "truth_column": "z_spec",
            "prediction_column": "z_pred",
        }
        d = build_report_data([m], PHOTOZ, "d")
        row = d["rows"][0]
        for key, value in evaluate_regression(pd.Series(t), p).items():
            self.assertAlmostEqual(row["metrics"][key], value)
        chart = row["slices"]["All objects"]["charts"]
        self.assertEqual(np.sum(chart["density"]["z"]), len(t))
        self.assertLess(len(chart["sample"]["x"]), len(t))


class EuclidReportTests(unittest.TestCase):
    def test_phosphoros_display_and_bootstrap_choose_same_valid_measurement(self) -> None:
        from photoz.data.euclid_report import (
            phosphoros_column,
            phosphoros_points,
            phosphoros_series,
        )

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "posterior.parquet"
            ids = np.array([2**60 + 1, 2**60 + 2])
            pd.DataFrame({
                "object_id": [ids[0], ids[0], ids[1]], "ppub_median": [-1, 0.5, -1],
                "ppub_lo68": [0, 0.4, 0], "ppub_hi68": [0, 0.6, 0],
            }).to_parquet(path)
            points, _ = phosphoros_points([path])
            native = phosphoros_series([path]).reindex(ids).to_numpy()
            local, _ = phosphoros_column(path, ids)
            np.testing.assert_equal(native, [points[int(ids[0])], np.nan])
            np.testing.assert_equal(local, native)

    def test_empty_scientific_comparisons_and_nonbracketing_ci_render(self) -> None:
        from photoz.bootstrap import arm_summary, tile_boot
        from photoz.euclid_report import write_json
        from photoz.figures import fig_baselines, fig_scatter

        empty = np.array([])
        interval = tile_boot(empty, None, empty, empty)
        self.assertTrue(np.isnan(interval).all())
        self.assertEqual(arm_summary({}, {}, empty, empty)["n"], 0)
        frame = pd.DataFrame({"z": empty, "ml": empty, "ppub": empty})
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            fig_scatter(folder, frame, "Empty", "scatter.pdf", 2)
            result = {"nmad": 0.1, "nmad_ci": [0.2, 0.3]}
            fig_baselines(folder, {"keys": [], "ALL": {
                "n": 2, "ml": result, "ppub": result, "poff": result,
            }})
            self.assertTrue((folder / "fig_baselines.pdf").is_file())
            write_json(folder / "facts.json", {"ci": interval}, as_float=True)
            self.assertEqual(json.loads((folder / "facts.json").read_text()), {"ci": [None, None]})

    def test_method_denominators_and_real_seed_identifiers(self) -> None:
        from photoz.data.euclid_report import ReportConfig, SeedStack
        from runtime.euclid_report import interactive_methods

        ids = np.array([2**56 + 1, 2**56 + 2], dtype=np.int64)
        values = np.array([[0.8, 1.2], [1.0, 1.4]])
        preds = {
            "north": SeedStack(ids[:1], values[:, :1], values[:, :1]),
            "nb": SeedStack(ids[1:], values[:, 1:], values[:, 1:]),
            "zval": SeedStack(ids, values, values),
            "valid": SeedStack(ids, values, values),
        }
        frame = pd.DataFrame(
            {
                "_set": "north + nb",
                "src": "DESI",
                "vis": [20, 23],
                "tile": [1, 2],
                "object_id": ids,
                "z": [1.0, 1.2],
                "ml": values.mean(0),
                "ppub": [0.9, np.nan],
                "poff": [1.0, 1.1],
                "zpf": [1.1, np.nan],
                "catboost": [0.95, 1.0],
            }
        )
        resource = {"hetdex": "f", "files": ["f"]}
        from photoz.data.euclid_report import PhosphorosSource

        cfg = ReportConfig.model_construct(
            tag="clip",
            seeds=[7, 9],
            learners=["catboost"],
            phosphoros=PhosphorosSource.model_validate(resource),
            phosphoros_zpf=PhosphorosSource.model_validate(resource),
        )
        val = {"ppub": np.array([0.9, 1.0]), "ml": values.mean(0)}
        methods = interactive_methods(
            cfg, frame, preds, {"clip_7": [{"epoch": 1, "loss": 0.2}]}, np.array([1.0, 1.2]), val,
            field_validation=frame.drop(columns=["_set", "src"]),
        )
        result = build_report_data(methods, PHOTOZ, "Euclid", strict=True)
        north = {r["name"]: r for r in result["rows"] if r["split"] == "N · north + nb"}
        self.assertEqual(north["phosphoros_public"]["coverage"]["requested"], 2)
        self.assertEqual(north["phosphoros_public"]["coverage"]["evaluated"], 1)
        self.assertEqual(north["CatBoost"]["coverage"]["evaluated"], 2)
        self.assertEqual(north["clip_7"]["details"]["seed"], 7)
        self.assertEqual(north["clip_7"]["history"][0]["loss"], 0.2)
        self.assertIn("euclid_phz", north)
        validation = {r["name"]: r for r in result["rows"] if r["split"] == "F · valid"}
        self.assertEqual(validation["CatBoost"]["role"], "development")
        self.assertEqual(validation["CatBoost"]["coverage"]["requested"], 2)
        self.assertTrue(result["overlaps"])


class OptionalAssetsTests(unittest.TestCase):
    def test_single_unlabeled_method_exposes_native_posteriors(self) -> None:
        ids = [2**60 + 9, 2**60 + 17]
        frame = pd.DataFrame(
            {"object_id": ids, "y_pred": [0.5, 0.7], "pdf": [[0.0, 1.0], [1.0, 2.0]]}
        )
        row = build_report_data(
            [
                {
                    "name": "SED",
                    "split": "catalog",
                    "truth_column": None,
                    "predictions": frame,
                    "posterior_grid": [0.0, 1.0],
                    "posterior_column": "pdf",
                }
            ],
            PHOTOZ,
            "native",
        )["rows"][0]
        self.assertEqual(row["coverage"]["evaluated"], 0)
        self.assertEqual(row["coverage"]["predicted"], 2)
        charts = row["slices"]["All objects"]["charts"]
        self.assertEqual(charts["posterior"]["count"], 0)
        self.assertEqual([x["id"] for x in charts["posterior_examples"]], [str(k) for k in ids])
        self.assertEqual(charts["posterior_examples"][0]["posterior"]["y"], [0.0, 1.0])

    def test_exact_cumulative_mass_selects_the_first_posterior_knot(self) -> None:
        from lib.report.examples import posterior_diagnostics

        method = {"posterior_grid": [0, 1, 2], "posterior_column": "pdf"}
        for weights in ([7, 2, 1], [70, 20, 10]):
            frame = pd.DataFrame({"_truth": [2.0], "pdf": [weights]})
            result = posterior_diagnostics(frame, method)
            assert result is not None
            index = result["nominal"].index(0.8)
            self.assertEqual(result["mean_width"][index], 1.0)
            self.assertEqual(result["empirical"][index], 0.0)
        tiny = np.nextafter(0.0, 1.0)
        frame = pd.DataFrame({"_truth": [2.0], "pdf": [[0.0, 0.0, tiny]]})
        result = posterior_diagnostics(frame, method)
        assert result is not None
        self.assertEqual(result["mean_width"], [0.0] * 5)

    def test_posteriors_thumbnail_and_standalone_export(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            Image.new("RGB", (25, 32), (50, 90, 110)).save(work / "sample.png")
            frame = pd.DataFrame(
                {
                    "object_id": [1, 2],
                    "y_true": [1.0, 2.0],
                    "y_pred": [1.0, 2.0],
                    "pdf": [[0.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
                    "stamp": ["sample.png", None],
                }
            )
            source = work / "pred.parquet"
            frame.to_parquet(source)
            options = {
                "predictions": source,
                "split": "test",
                "posterior_column": "pdf",
                "posterior_grid": [0, 1, 2],
                "thumbnail_column": "stamp",
            }
            output = render_report(
                work, [dict(options, name="A"), dict(options, name="B")], PHOTOZ, "PDF"
            )
            all_rows = output["rows"][0]["slices"]["All objects"]
            calibration = all_rows["charts"]["posterior"]
            self.assertEqual(calibration["count"], 1)
            self.assertEqual(calibration["invalid"], 1)
            self.assertEqual(calibration["empirical"], [1.0] * 5)
            self.assertEqual(calibration["mean_width"], [0.0] * 5)
            example = output["pairs"][0]["slices"]["All objects"]["examples"][0]
            self.assertEqual(example["assets_a"]["posterior"]["y"], [0.0, 1.0, 0.0])
            self.assertTrue(example["assets_b"]["thumbnail"].startswith("data:image/jpeg;base64,"))
            source.unlink()
            (work / "sample.png").unlink()
            from lib.report import write_report

            write_report(work / "replay", json.loads((work / "report-data.json").read_text()))
            for file in ("report-data.json", "metrics.json", "metrics.csv", "report.html"):
                self.assertEqual((work / file).read_bytes(), (work / "replay" / file).read_bytes())


class HistoricalReportTests(unittest.TestCase):
    def test_existing_reports_cover_extra_metrics_and_normalized_threshold(self) -> None:
        from photoz.legacy_plots import redshift_scatter
        from runtime.render_table import Row, all_metrics, best_map, is_best

        rows = [
            Row("a", {"r2": 0.2, "mae": 1.0, "bias": -0.2, "custom": 7.0}),
            Row("b", {"r2": 0.8, "mae": 0.5, "bias": 0.09}),
        ]
        keys = all_metrics(rows)
        self.assertEqual(set(keys), set(rows[0].test))
        self.assertEqual(best_map(rows, keys), {"mae": 0.5, "r2": 0.8, "bias": 0.09})
        self.assertTrue(is_best(0.8, 0.8, "r2"))
        self.assertFalse(is_best(0.2, 0.8, "r2"))
        self.assertFalse(is_best(-0.2, 0.09, "bias"))
        fig = redshift_scatter(pd.Series([0.0, 1.0]), np.array([0.0, 1.0]), "sample", 0.0, 1.0)
        self.assertEqual(float(fig.data[-2].y[-1]), 1.3)
        self.assertEqual(float(fig.data[-1].y[-1]), 0.7)
