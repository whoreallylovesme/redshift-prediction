import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import joblib
import numpy as np
import pandas as pd
import yaml
from prefect.testing.utilities import prefect_test_harness
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from lib.config.experiment import Dataset
from lib.models.tabular.factories import catboost
from lib.workflow.data import prepare_dataset
from runtime.pipeline import pipeline


class TabularRecipeTests(TestCase):
    def test_explicit_features_and_chronological_split(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.parquet"
            pd.DataFrame({"sample": range(20), "x": range(20), "response": range(20)}).to_parquet(
                source
            )
            dataset = Dataset(
                source=source,
                id_column="sample",
                target="response",
                features=["x"],
                shuffle=False,
                test_size=0.35,
            )
            data = prepare_dataset(dataset, root / "parts")
            self.assertEqual(data["features"], ["x"])
            self.assertEqual(pd.read_parquet(data["test"])["sample"].tolist(), list(range(13, 20)))
            bad = dataset.model_copy(update={"features": ["unknown", "x"]})
            with self.assertRaisesRegex(ValueError, "unknown"):
                prepare_dataset(bad, root / "invalid")
            self.assertFalse((root / "invalid").exists())
            held = root / "held.parquet"
            pd.DataFrame({"sample": [100], "response": [1]}).to_parquet(held)
            separate = Dataset(
                source=source,
                features=["x"],
                id_column="sample",
                target="response",
                evaluation_files={"validation": held},
            )
            with self.assertRaisesRegex(ValueError, "validation.*'x'"):
                prepare_dataset(separate, root / "invalid")
            self.assertFalse((root / "invalid").exists())
            with self.assertRaisesRegex(ValueError, "split_column"):
                Dataset(source=source, split_column="part", test_size=0.25)

    def test_only_named_estimators_are_accepted(self) -> None:
        from lib.config.experiment import Tabular
        from lib.utils.estimator_loader import load_estimator

        code = "from sklearn.linear_model import Ridge\nestimator = Ridge()"
        with self.assertRaisesRegex(ValueError, "Unknown estimator"):
            Tabular.model_validate({"kind": "tabular", "model": code})
        with self.assertRaisesRegex(ValueError, "Unknown estimator"):
            load_estimator(code)

    def test_catboost_weighted_grid_matches_independent_fit(self) -> None:
        rng = np.random.default_rng(27)
        x = rng.normal(size=(45, 3))
        frame = pd.DataFrame(x, columns=list("abc"))
        frame["y"] = x[:, 0] + 2 * x[:, 1] + rng.normal(size=45) * 1.8
        frame["w"] = np.where(frame.y > 2, 50.0, 1.0)
        frame["id"] = np.arange(len(frame))
        with TemporaryDirectory() as temporary, prefect_test_harness():
            root = Path(temporary)
            path = root / "source.parquet"
            frame.to_parquet(path, index=False)
            flow = root / "study.yaml"
            flow.write_text(
                yaml.safe_dump(
                    {
                        "name": "weighted tree search",
                        "output": str(root / "runs"),
                        "cache": False,
                        "dataset": {
                            "source": str(path),
                            "features": list("abc"),
                            "target": "y",
                            "id_column": "id",
                            "weight_column": "w",
                        },
                        "methods": {
                            "tree": {
                                "kind": "tabular",
                                "model": "catboost",
                                "params": {"iterations": 8},
                                "train": {
                                    "split": {"val_size": 0.1, "random_seed": 42},
                                    "hyperopt": {"enabled": True, "cv": 2, "verbose": 0},
                                    "param_grid": {"model__depth": [2, 5]},
                                },
                            }
                        },
                    }
                )
            )
            result = pipeline(str(flow))
            method = result / "methods/tree"
            fitted = joblib.load(method / "model.pkl")
            report = json.loads((method / "train_report.json").read_text())
            train = pd.read_parquet(result / "data/train.parquet")
            xt, xv, yt, _ = train_test_split(
                train[list("abc")], train.y, test_size=0.1, shuffle=True, random_state=42
            )
            weights = train.w.loc[xt.index]
            losses = []
            for depth in (2, 5):
                scores = []
                for ti, vi in KFold(2).split(xt):
                    reference = catboost().set_params(
                        model__iterations=8, model__depth=depth, model__thread_count=1
                    )
                    reference.fit(xt.iloc[ti], yt.iloc[ti], model__sample_weight=weights.iloc[ti])
                    scores.append(
                        root_mean_squared_error(
                            yt.iloc[vi],
                            reference.predict(xt.iloc[vi]),
                            sample_weight=weights.iloc[vi],
                        )
                    )
                losses.append(np.mean(scores))
            selected = [2, 5][int(np.argmin(losses))]
            self.assertEqual(report["best_params"], {"model__depth": selected})
            reference = catboost().set_params(
                model__iterations=8, model__depth=selected, model__thread_count=1
            )
            reference.fit(xt, yt, model__sample_weight=weights)
            np.testing.assert_array_equal(reference.predict(xv), fitted.predict(xv))
