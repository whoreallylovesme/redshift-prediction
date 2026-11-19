import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from lib.config.schemas import FlowConfig
from photoz import legacy_flow
from photoz.data.hdf5 import HDF5Dataset, get_hdf5_dataloaders
from photoz.models.capsnet import CapsNet
from runtime.train import train


class CapsNetStagesTest(TestCase):
    def test_repeated_reports_use_only_epochs_appended_by_each_fit(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            log = root / "log.jsonl"
            old = {"stage": "B", "epoch": 9}
            log.write_text(json.dumps(old) + "\n")
            recipe = root / "recipe.yaml"
            recipe.write_text(
                yaml.safe_dump(
                    {
                        "out_dir": str(root),
                        "inference": {"out": str(root / "predictions.parquet")},
                    }
                )
            )
            config = FlowConfig(pipeline="capsnet", runs_root=temp, train={"config": str(recipe)})
            histories = [
                [{"stage": "A", "epoch": 1}, {"stage": "B", "epoch": 1}],
                [{"stage": "B", "epoch": 1}],
            ]
            with (
                patch.object(legacy_flow, "fit_capsnet") as fit,
                patch.object(legacy_flow, "predict_capsnet_splits"),
                patch.object(legacy_flow, "evaluate_capsnet_split"),
                patch.object(legacy_flow, "combined_report") as report,
            ):
                batches = iter(histories)

                def append_epochs(*_: Any) -> None:
                    with log.open("a") as stream:
                        for epoch in next(batches):
                            stream.write(json.dumps(epoch) + "\n")

                fit.side_effect = append_epochs
                for epochs in histories:
                    legacy_flow.run_capsnet_pipeline(config)
                    [row] = report.call_args.args[1]
                    self.assertEqual(row["history"], epochs)
            self.assertEqual(
                [json.loads(line) for line in log.read_text().splitlines()],
                [old, *histories[0], *histories[1]],
            )

    def test_stage_b_continues_optimizer_without_training_on_validation_objects(self) -> None:
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            images, morph = root / "objects.h5", root / "morph.parquet"
            n = 12
            rng = np.random.default_rng(8)
            with h5py.File(images, "w") as data:
                data["images"] = rng.normal(size=(n, 5, 16, 16)).astype("f4")
                data["specz_redshift"] = np.linspace(0.03, 0.36, n)
            labels = (np.arange(n) % 2).astype(float)
            labels[[1, 5, 7]] = np.nan
            pd.DataFrame({"gz_morpho_binary": labels}).to_parquet(morph)
            args: dict[str, Any] = dict(
                h5_path=images,
                morph_parquet=morph,
                batch_size=16,
                num_workers=0,
                train_ratio=0.7,
                weighted_sampling=False,
                seed=9,
            )
            a_train, a_val = get_hdf5_dataloaders(**args, labeled_only=True)
            b_train, b_val = get_hdf5_dataloaders(**args)
            for a, b in ((a_train, b_train), (a_val, b_val)):
                assert isinstance(a.dataset, HDF5Dataset) and isinstance(b.dataset, HDF5Dataset)
                a_ids, b_ids = a.dataset.indices, b.dataset.indices
                np.testing.assert_array_equal(a_ids, b_ids[np.isfinite(labels[b_ids])])
            cfg = dict(
                model_type="lightning",
                model="capsnet",
                crop_size=16,
                device="cpu",
                stage_a_epochs=1,
                stage_b_epochs=1,
                out_dir=str(root / "out"),
                **{**args, "h5_path": str(images), "morph_parquet": str(morph)},
            )
            recipe = root / "stages.yaml"
            recipe.write_text(yaml.safe_dump(cfg))
            train(str(recipe))
            weights = root / "out/weights"
            steps = []
            for tag in ("a", "b"):
                ckpt = torch.load(weights / f"stage_{tag}_best.ckpt", weights_only=True)
                opt = ckpt["optimizer_states"][0]
                steps.append(max(int(state["step"]) for state in opt["state"].values()))
                restored = CapsNet(input_size=16)
                for kind in ("best", "last"):
                    state = torch.load(weights / f"stage_{tag}_{kind}.pt", weights_only=True)
                    restored.load_state_dict(state, strict=True)
                for name, parameter in restored.state_dict().items():
                    torch.testing.assert_close(parameter, ckpt["state_dict"]["net." + name])
            self.assertEqual(steps, [1, 2])
            hist = [json.loads(line) for line in (root / "out/log.jsonl").read_text().splitlines()]
            self.assertEqual([r["stage"] for r in hist], ["A", "B"])
            self.assertAlmostEqual(hist[1]["lr"], hist[0]["lr"] * 0.95)

            pd.DataFrame({"gz_morpho_binary": np.full(n, np.nan)}).to_parquet(morph)
            cfg.update(stage_a_epochs=0, out_dir=str(root / "b"))
            recipe.write_text(yaml.safe_dump(cfg))
            train(str(recipe))
            self.assertTrue((root / "b/weights/stage_b_best.pt").is_file())
            self.assertFalse((root / "b/weights/stage_a_last.pt").exists())
