import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from lib.config.experiment import Dataset
from lib.workflow.data import prepare_dataset
from lib.workflow.experiment import fit, predict_table, save_tabular_model


class NamedTablesTest(TestCase):
    def test_external_splits_share_model_and_allow_missing_truth(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "a.csv"
            train.write_text("id,x,y\na,0,0\nb,1,2\nc,2,4\n")
            b, c = root / "b.csv", root / "c.csv"
            b.write_text("id,x,y\n001,4,8\n")
            c.write_text("id,x\n006,2\n")
            cfg = Dataset(
                source=train,
                target="y",
                features=["x"],
                id_column="id",
                evaluation_files={"B": b, "C": c},
            )
            data = prepare_dataset(cfg, root / "data")
            recipe = root / "model.yaml"
            recipe.write_text("estimator: ridge\nsplit: {val_size: 0}\n")
            fitted = fit.fn(data, recipe, "regression", 1, None)
            model = save_tabular_model.fn(fitted, root / "method")
            self.assertEqual(
                json.loads((model.parent / "train_report.json").read_text())["train_rows"], 3
            )
            for split, src in data["evaluation_files"].items():
                out = root / f"pred_{split}.parquet"
                predict_table.fn(model, src, data["features"], data["target"], out)
                pred = pd.read_parquet(out)
                self.assertEqual(pred.id.iloc[0], "001" if split == "B" else "006")
                self.assertEqual("y_true" in pred, split == "B")
            c.write_text("id,x\na,4\n")
            with self.assertRaisesRegex(ValueError, "shares identifiers"):
                prepare_dataset(cfg, root / "collision")
