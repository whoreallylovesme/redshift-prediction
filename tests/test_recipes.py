from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import pandas as pd
import yaml
from click.testing import CliRunner
from hydra import compose, initialize_config_dir

from lib.config import PytorchTrainConfig, load_flow
from lib.config.experiment import Dataset, Experiment
from lib.utils.hydra_utils import hydra_to_yaml
from lib.utils.utils import load_yaml_as
from lib.workflow.data import prepare_dataset
from runtime.pipeline import main


class RecipesTest(TestCase):
    def test_config_selection_and_render_without_a_flow(self) -> None:
        runner = CliRunner()
        with patch("runtime.pipeline._run") as run:
            for args in ([], ["--check"], ["workers=2"]):
                result = runner.invoke(main, args)
                self.assertEqual(result.exit_code, 2, result.output)
                self.assertIn("--config", result.output)
            with (
                TemporaryDirectory() as temp,
                patch("lib.workflow.experiment.rebuild_report") as rebuild,
            ):
                result = runner.invoke(main, ["--render", temp])
                self.assertEqual(result.exit_code, 0, result.output)
                rebuild.assert_called_once_with(Path(temp))
            run.assert_not_called()

    def test_all_shipped_flows(self) -> None:
        flows = sorted(Path("examples/configs").glob("*.yaml"))
        self.assertEqual({p.stem for p in flows}, {"teddy", "euclid"})
        for path in flows:
            with self.subTest(path=str(path)):
                load_flow(path, check=True)

    def test_check_and_execution_compose_the_same_flow(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "methods").mkdir()
            (root / "methods" / "saved.yaml").write_text(
                "reference: {kind: saved, predictions: held.parquet}\n"
            )
            (root / "methods" / "alternate.yaml").write_text(
                "alternate: {kind: saved, predictions: alternate.parquet}\n"
            )
            flow = root / "study.yaml"
            flow.write_text(
                "defaults: [ {methods: saved}, _self_ ]\nname: trial\noutput: runs/${name}\n"
            )
            runner = CliRunner()
            with patch("runtime.pipeline._run") as run:
                run.return_value = None
                args = ["--config", str(flow), "--workers", "3", "methods=alternate", "name=demo"]
                checked = runner.invoke(main, [*args, "--check"])
                self.assertEqual(checked.exit_code, 0, checked.output)
                run.assert_not_called()
                launched = runner.invoke(main, args)
                self.assertEqual(launched.exit_code, 0, launched.output)
                actual = run.call_args.args[0]
            self.assertIsInstance(actual, Experiment)
            self.assertEqual(actual.output, Path("runs/demo"))
            self.assertEqual(actual.workers, 3)
            self.assertIn("alternate", actual.methods)
            self.assertEqual(
                actual,
                load_flow(
                    flow, check=True, workers=3, overrides=("methods=alternate", "name=demo")
                ),
            )

    def test_check_reports_reference_and_schema_errors_without_training(self) -> None:
        runner = CliRunner()
        sample = {
            "name": "check", "output": "runs/check",
            "dataset": {"source": "input.csv", "features": ["x"], "target": "y"},
        }
        with TemporaryDirectory() as temp:
            root = Path(temp)
            recipe = root / "model.yaml"
            recipe.write_text("estimator: misspelled_model\n")
            for method, error in [
                ({"kind": "tabular", "recipe": str(recipe)}, "misspelled_model"),
                ({"kind": "external", "recipe": str(recipe), "predictor": "math:pi"}, "callable"),
                ({"kind": "external", "recipe": str(recipe), "predictor": "math:typo"}, "typo"),
                ({"kind": "tabular", "recipe": str(root / "missing.yaml")}, "missing.yaml"),
            ]:
                with self.subTest(error=error):
                    flow = root / "flow.yaml"
                    flow.write_text(yaml.safe_dump({**sample, "methods": {"bad": method}}))
                    result = runner.invoke(main, ["--config", str(flow), "--check"])
                    self.assertNotEqual(result.exit_code, 0)
                    self.assertIn(error, result.output)

    def test_plain_flow_inside_active_hydra_and_image_only_feature_validation(self) -> None:
        with initialize_config_dir(
            config_dir=str(Path("examples/configs").resolve()), version_base="1.3"
        ):
            config = load_flow(Path("examples/configs/teddy.yaml"), check=True)
            self.assertIsInstance(config, Experiment)
        image = yaml.safe_load(Path("examples/configs/teddy.yaml").read_text())
        del image["methods"]["catboost"]
        image["dataset"]["features"] = [image["dataset"]["target"]]
        with self.assertRaisesRegex(ValueError, "exclude target"):
            Experiment.model_validate(image, context={"check": True})

    def test_hydra_preserves_ids_and_training_overrides(self) -> None:
        folder = Path("tests/fixtures/legacy/hydra").resolve()
        with initialize_config_dir(config_dir=str(folder), version_base="1.3"):
            cfg = compose("hydra", overrides=["model=fusion", "dataset=synthetic", "train.lr=0.3"])
            path = hydra_to_yaml(cfg, folder)
            try:
                recipe = load_yaml_as(path, PytorchTrainConfig)
                self.assertEqual(recipe.lr, 0.3)
                self.assertIsNotNone(recipe.image_catalog)
                assert recipe.image_catalog is not None
                self.assertEqual(recipe.image_catalog.id_column, "sample")
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
