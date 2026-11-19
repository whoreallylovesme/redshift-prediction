import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase

import h5py
import numpy as np
import pandas as pd
import torch
import yaml

from lib.models.nn.capsnet import CapsNet
from photoz.data.hayat import HayatDataset, get_hayat_dataloaders
from photoz.training.capsnet import run_train


class CapsNetStagesTest(TestCase):
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
            a_train, a_val = get_hayat_dataloaders(**args, labeled_only=True)
            b_train, b_val = get_hayat_dataloaders(**args)
            for a, b in ((a_train, b_train), (a_val, b_val)):
                assert isinstance(a.dataset, HayatDataset) and isinstance(b.dataset, HayatDataset)
                a_ids, b_ids = a.dataset.indices, b.dataset.indices
                np.testing.assert_array_equal(a_ids, b_ids[np.isfinite(labels[b_ids])])
            cfg = dict(
                crop_size=16,
                device="cpu",
                stage_a_epochs=1,
                stage_b_epochs=1,
                out_dir=str(root / "out"),
                **{**args, "h5_path": str(images), "morph_parquet": str(morph)},
            )
            recipe = root / "stages.yaml"
            recipe.write_text(yaml.safe_dump(cfg))
            run_train(recipe)
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
            run_train(recipe)
            self.assertTrue((root / "b/weights/stage_b_best.pt").is_file())
            self.assertFalse((root / "b/weights/stage_a_last.pt").exists())
