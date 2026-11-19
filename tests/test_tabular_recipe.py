from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from lib.config.experiment import Dataset
from lib.models.factories import catboost
from lib.workflow.data import prepare_dataset
from lib.workflow.experiment import fit


class TabularRecipeTests(TestCase):
    def test_legacy_split_file_and_partial_feature_sets(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame(
                {
                    "sample": list(range(20)),
                    "x": list(range(20)),
                    "constant_text": ["column"] * 20,
                    "response": list(range(20)),
                }
            ).to_parquet(root / "input.parquet")
            (root / "columns.yaml").write_text(
                yaml.safe_dump(
                    {
                        "target": "response",
                        "feature_sets": {"old": ["absent", "x", "constant_text"]},
                    }
                )
            )
            (root / "split.yaml").write_text(
                yaml.safe_dump(
                    {
                        "split": {"shuffle": False, "test_size": 0.35, "random_seed": 22},
                    }
                )
            )
            data = prepare_dataset(
                Dataset(
                    source=root / "input.parquet",
                    id_column="sample",
                    feature_catalog=root / "columns.yaml",
                    feature_set="old",
                    split_config=root / "split.yaml",
                ),
                root / "parts",
            )
            self.assertEqual(data["features"], ["x"])
            self.assertEqual(data["missing_features"], ["absent", "constant_text"])
            self.assertEqual(pd.read_parquet(data["test"])["sample"].tolist(), list(range(13, 20)))

    def test_catboost_weighted_grid_matches_independent_fit(self) -> None:
        rng = np.random.default_rng(27)
        x = rng.normal(size=(45, 3))
        frame = pd.DataFrame(x, columns=list("abc"))
        frame["y"] = x[:, 0] + 2 * x[:, 1] + rng.normal(size=45) * 1.8
        frame["w"] = np.where(frame.y > 2, 50.0, 1.0)
        frame["id"] = np.arange(len(frame))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "source.parquet"
            frame.to_parquet(path, index=False)
            data = prepare_dataset(
                Dataset(
                    source=path,
                    features=list("abc"),
                    id_column="id",
                    target="y",
                    weight_column="w",
                ),
                root / "parts",
            )
            recipe = root / "recipe.yaml"
            recipe.write_text(
                yaml.safe_dump(
                    {
                        "estimator": "catboost",
                        "estimator_params": {"model__iterations": 8},
                        "hyperopt": {"enabled": True, "cv": 2, "verbose": 0},
                        "param_grid": {"model__depth": [2, 5]},
                    }
                )
            )
            fitted, report = fit.fn(data, recipe, "regression", 1, None)
            train = pd.read_parquet(data["train"])
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
