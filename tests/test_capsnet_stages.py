import json
from collections.abc import Sized
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from unittest import TestCase
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from lib.models.nn.capsnet import CapsNet
from photoz.data.hayat import HayatDataset, get_hayat_dataloaders
from photoz.data.teddy_capsnet import staged_capsnet_loaders
from photoz.training.capsnet import run_train


def inference_fixture() -> tuple[pd.DataFrame, DataLoader]:
    values = torch.zeros(2, 5, 16, 16)
    targets = torch.zeros(2)
    loader = DataLoader(TensorDataset(values, targets, targets), batch_size=2)
    return pd.DataFrame({"objid": [2, 1], "z_spec": [0.1, 0.2]}), loader


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
                data={"_target_": "photoz.data.hayat.staged_capsnet_loaders"},
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


    def test_teddy_factory_builds_both_stages(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / "images"
            images.mkdir()
            rows = []
            for index in range(12):
                np.savez(images / f"{index}.npz", image=np.ones((16, 16, 4), dtype=np.float32))
                rows.append(f"{index} 0 0 0 0 0 {0.03 + index * 0.02}")
            catalog = root / "catalog.txt"
            catalog.write_text("\n".join(rows))
            morph = root / "morph.parquet"
            pd.DataFrame({
                "objid": [str(i) for i in range(12)],
                "gz_spiral": [i % 2 for i in range(12)],
                "gz_elliptical": [(i + 1) % 2 for i in range(12)],
            }).to_parquet(morph)
            loaders = staged_capsnet_loaders(
                images,
                catalog,
                morph,
                batch_size=4,
                num_workers=0,
                train_ratio=0.75,
                seed=9,
                crop_size=16,
                in_channels=4,
            )
            assert loaders.train_a is not None and loaders.val_a is not None
            self.assertEqual(
                len(cast(Sized, loaders.train_a.dataset))
                + len(cast(Sized, loaders.val_a.dataset)),
                12,
            )
            self.assertEqual(
                len(cast(Sized, loaders.train_b.dataset))
                + len(cast(Sized, loaders.val_b.dataset)),
                12,
            )
            for loader in (loaders.train_a, loaders.val_a, loaders.train_b, loaders.val_b):
                image, _, _ = next(iter(loader))
                self.assertEqual(image.shape[1:], (4, 16, 16))


    def test_inference_uses_configured_data_factory(self) -> None:
        from photoz.methods import capsnet as method

        with TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "predictions.parquet"
            recipe = root / "inference.yaml"
            recipe.write_text(yaml.safe_dump({
                "out_dir": str(root),
                "inference": {
                    "out": str(output),
                    "data": {"_target_": "test_capsnet_stages.inference_fixture"},
                },
            }))
            predictions = {
                "z_pred": np.array([0.11, 0.21]),
                "p_spiral": np.array([0.4, 0.6]),
            }
            with (
                patch.object(method, "build_predictor", return_value=object()),
                patch.object(method, "_predict", return_value=predictions),
            ):
                method.run_inference(recipe)
            result = pd.read_parquet(output)
            self.assertEqual(result.objid.tolist(), [2, 1])
            np.testing.assert_allclose(result.z_pred, predictions["z_pred"])
            np.testing.assert_allclose(result.p_spiral, predictions["p_spiral"])
