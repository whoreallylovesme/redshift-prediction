from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from lib.config.experiment import Dataset
from lib.config.schemas import ImageCatalogConfig, PytorchTrainConfig
from lib.dataloaders.images import catalog
from lib.train.regression_lightning import RegressionModule
from lib.workflow.data import prepare_dataset


class RegressionIntegrityTests(TestCase):
    def test_table_ids_survive_cleaning_and_export(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.csv"
            frame = pd.DataFrame(
                {
                    "object_id": ["0001", "0002", "0003", "0004"],
                    "x": [1.0, 2.0, 3.0, 4.0],
                    "response": [4, 3, 2, 1],
                }
            )
            frame.to_csv(source, index=False)
            cfg = Dataset(
                source=source,
                features=["x"],
                target="response",
                id_column="object_id",
                cleaner="dummy:clean",
            )
            with self.assertRaisesRegex(ValueError, "prediction columns"):
                Dataset(source=source, features=["x"], target="response", id_column="y_pred")

            def duplicate(items: pd.DataFrame, target: str) -> pd.DataFrame:
                return pd.concat([items, items.iloc[:1]], ignore_index=True)

            with patch("lib.workflow.data.plugin", return_value=duplicate):
                with self.assertRaisesRegex(ValueError, "unique"):
                    prepare_dataset(cfg, root / "output")
            cfg.cleaner = None
            report = prepare_dataset(cfg, root / "good")
            ids = pd.concat([pd.read_parquet(report[k]).object_id for k in ["train", "test"]])
            self.assertEqual(set(ids), set(frame.object_id))

    def test_image_id_coercion_never_rounds_or_wraps(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "instances.parquet"
            spec = ImageCatalogConfig(id_column="id", target_column="value", id_dtype="int64")
            for value in [pd.Series([1.9]), pd.Series([2**64 - 1], dtype="uint64")]:
                pd.DataFrame({"id": value}).to_parquet(path)
                with self.assertRaises((ValueError, OverflowError)):
                    catalog(path, spec)
            source = root / "catalog.csv"
            source.write_text("id,value\n001,3\n010,4\n")
            self.assertEqual(
                catalog(source, spec.model_copy(update={"id_dtype": None})).id.tolist(),
                ["001", "010"],
            )

    def test_singleton_prediction_and_target_cannot_broadcast(self) -> None:
        cfg = PytorchTrainConfig.model_construct(freeze_epochs=0, label_noise=0)
        module = RegressionModule(cfg, torch.nn.Identity())
        _, first, _ = module._step((torch.tensor([1.0, 2.0]), torch.tensor([[1.0], [2.0]])))
        _, last, _ = module._step((torch.tensor([3.0]), torch.tensor([3.0])))
        self.assertEqual(np.concatenate([first.numpy(), last.numpy()]).tolist(), [1.0, 2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "one scalar"):
            module._step((torch.ones(2, 2), torch.ones(2)))

    def test_mixed_csv_and_parquet_id_types_cannot_bypass_training_embargo(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            training = root / "train.csv"
            training.write_text("id,x,y\n7,1,2\n9,2,4\n")
            held = root / "held.parquet"
            pd.DataFrame({"id": [7], "x": [1], "y": [2]}).to_parquet(held)
            cfg = Dataset(
                source=training,
                id_column="id",
                features=["x"],
                target="y",
                evaluation_files={"N": held},
            )
            with self.assertRaisesRegex(ValueError, "different ID type"):
                prepare_dataset(cfg, root / "bad")
            cfg.csv_id_type = "int64"
            with self.assertRaisesRegex(ValueError, "shares identifiers"):
                prepare_dataset(cfg, root / "still-bad")

    def test_reference_intervals_cannot_replace_prediction_intervals(self) -> None:
        from lib.report import build_report_data

        pred = pd.DataFrame(
            {"id": ["1", "2"], "y_pred": [1.0, 2.0], "lo": [0.5, 1.5], "hi": [1.5, 2.5]}
        )
        ref = pd.DataFrame(
            {"id": ["1", "2"], "y_true": [1.0, 2.0], "lo": [-3.0, -3.0], "hi": [-1.0, -1.0]}
        )
        method = {
            "name": "m",
            "split": "N",
            "predictions": pred,
            "reference": ref,
            "id_column": "id",
            "lower_column": "lo",
            "upper_column": "hi",
        }
        result = build_report_data([method], "regression", "test", strict=True)
        self.assertEqual(
            result["rows"][0]["slices"]["All objects"]["charts"]["interval"]["covered"], 2
        )
