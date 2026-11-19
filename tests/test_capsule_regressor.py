from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd
import torch

from lib.config import PytorchTrainConfig
from lib.inference.images import predict_catalog
from lib.models.nn.capsnet import CapsNet
from lib.models.nn.factories import capsnet
from lib.train.regression_lightning import fit_images


class CapsuleRegressorTest(TestCase):
    def test_generic_one_channel_fit_predict_and_native_state_compatibility(self) -> None:
        previous = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            ids = [f"s{i}" for i in range(8)]
            target = np.linspace(-0.5, 0.5, len(ids))
            rng = np.random.default_rng(9)
            for i, name in enumerate(ids):
                np.savez(
                    images / f"{name}.npz", image=rng.normal(target[i], 0.2, (16, 16)).astype("f4")
                )
            frame = pd.DataFrame({"sensor": ids, "temperature": target})
            frame.iloc[:6].to_csv(root / "train.csv", index=False)
            frame.iloc[6:].to_csv(root / "test.csv", index=False)
            params = dict(image_channels=1, input_size=16, num_class=3, routings=2, dim_capsule=8)
            out = root / "run"
            cfg = PytorchTrainConfig.model_validate({
                "model_type": "lightning", "device": "cpu", "estimator": "capsnet",
                "estimator_params": params, "out_size": 16, "data_dir": str(images),
                "catalog": str(root / "train.csv"),
                "image_catalog": {"id_column": "sensor", "target_column": "temperature"},
                "epochs": 1, "batch_size": 3, "train_ratio": 0.5, "num_workers": 0, "lr": 0.0005,
            })
            weights = fit_images(cfg, out)
            predict_catalog(
                cfg, cfg.image_catalog, weights, images, root / "test.csv",
                out / "preds.parquet", "sensor", held_out=True,
            )
            pred = pd.read_parquet(out / "preds.parquet")
            self.assertEqual(pred.sensor.tolist(), ids[6:])
            np.testing.assert_allclose(pred.y_true, target[6:])
            self.assertTrue(np.isfinite(pred.y_pred).all())

            native = CapsNet(
                in_channels=1, input_size=16, num_class=3, routings=2, dim_capsule=8
            ).eval()
            native.load_state_dict(
                torch.load(out / "best_model.pt", weights_only=True), strict=True
            )
            scalar = capsnet(**params).eval()
            scalar.load_state_dict(native.state_dict(), strict=True)
            with np.load(out / "image_normalization.npz") as stats:
                normalized = []
                for name in ids[6:]:
                    with np.load(images / f"{name}.npz") as stamp:
                        normalized.append(
                            (stamp["image"] - stats["mean"][0]) / (stats["std"][0] + 1e-8)
                        )
            x = torch.from_numpy(np.stack(normalized)[:, None])
            with torch.inference_mode():
                value, classes, reconstruction = native(x)
                torch.testing.assert_close(scalar(x), value, rtol=0, atol=0)
            self.assertEqual(classes.shape, (2, 3))
            self.assertEqual(reconstruction.shape, x.shape)
            np.testing.assert_allclose(pred.y_pred, value.numpy(), rtol=1e-6, atol=1e-6)
