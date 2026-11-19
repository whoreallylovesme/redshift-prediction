from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd
from hydra import compose, initialize_config_dir

from lib.config import PytorchTrainConfig
from lib.config.check import check_recipe, mapping
from lib.config.experiment import Dataset
from lib.utils.hydra_utils import hydra_to_yaml
from lib.workflow.data import prepare_dataset


class RecipesTest(TestCase):
    def test_all_shipped_flows(self) -> None:
        for path in sorted(Path("configs/flows").glob("*.yaml")):
            with self.subTest(path=str(path)):
                check_recipe(path)

    def test_hydra_preserves_ids_and_training_overrides(self) -> None:
        with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base="1.3"):
            for model in ("capsnet", "convnext", "vit"):
                cfg = compose(
                    "hydra", overrides=[f"model={model}", "dataset=teddy", "train.lr=0.3"]
                )
                path = hydra_to_yaml(cfg)
                try:
                    raw = mapping(path)
                    if model == "capsnet":
                        self.assertEqual((raw["dataset"], raw["lr"]), ("teddy", 0.3))
                    else:
                        recipe = PytorchTrainConfig.model_validate(raw)
                        self.assertEqual(recipe.lr, 0.3)
                        self.assertEqual(recipe.data_dir, "teddyA_imgs")
                finally:
                    path.unlink()

    def test_text_csv_ids_and_no_generated_id_collisions_across_sources(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "raw.csv"
            path.write_text("id,x,y,fold\n0001,5,8,train\n0004,7,9,test\n")
            cfg = Dataset(
                source=path, id_column="id", features=["x"], target="y", split_column="fold"
            )
            data = prepare_dataset(cfg, root / "split")
            self.assertEqual(pd.read_parquet(data["test"]).id.tolist(), ["0004"])
            auto = cfg.model_copy(update={"id_column": None})
            a = prepare_dataset(auto, root / "a")
            path.write_text(path.read_text().replace("0004", "1234"))
            b = prepare_dataset(auto, root / "b")
            self.assertNotEqual(
                pd.read_parquet(a["test"])["__source_row"].iloc[0],
                pd.read_parquet(b["test"])["__source_row"].iloc[0],
            )
