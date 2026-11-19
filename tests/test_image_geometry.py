from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as torch_f

from lib.config.schemas import ImageCatalogConfig
from lib.dataloaders.common import center_crop, random_rot90_flip
from lib.dataloaders.images import Images
from photoz.data.npz import PhotozDataset, _center_crop_resize, load_npz


def old_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    return img[top : top + size, left : left + size]


def old_resize(img: np.ndarray, size: int) -> torch.Tensor:
    chw = torch.from_numpy(img).permute(2, 0, 1)
    return torch_f.interpolate(
        chw[None].float(),
        size=[size, size],
        mode="bilinear",
        align_corners=False,
    )[0]


class ImageGeometryTests(TestCase):
    def test_catalog_and_teddy_keep_their_own_normalization_order(self) -> None:
        rng = np.random.default_rng(867)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for channels in (1, 4, 6):
                image = rng.normal(size=(11, 13, channels)).astype("float32")
                np.savez(root / "01.npz", image=image if channels != 1 else image[:, :, 0])
                rows = pd.DataFrame({"id": ["01"], "target": [0.4]})
                stats = (
                    rng.normal(size=channels).astype("f4"),
                    rng.uniform(1, 2, channels).astype("f4"),
                )
                for size in (4, 8, 14):
                    with self.subTest(channels=channels, size=size):
                        crop = old_crop(image, 8)
                        from_teddy = _center_crop_resize(image, 8, size)
                        expected = old_resize(crop, size).permute(1, 2, 0).numpy()
                        np.testing.assert_array_equal(from_teddy, expected)
                        self.assertEqual(from_teddy.dtype, image.dtype)
                        spec = ImageCatalogConfig(id_column="id", target_column="target")
                        observed, _ = Images(root, rows, spec, stats, 8, size)[0]
                        scaled = (crop - stats[0]) / (stats[1] + 1e-8)
                        self.assertIsInstance(observed, torch.Tensor)
                        torch.testing.assert_close(
                            observed, old_resize(scaled, size), rtol=0, atol=0
                        )
            raw = rng.integers(0, 255, size=(7, 12), dtype="uint8")[::-1]
            np.testing.assert_array_equal(center_crop(raw, 6), old_crop(raw, 6))
            np.testing.assert_array_equal(center_crop(raw.T, 10), old_crop(raw.T, 10))
            native_stamps = raw.astype(">u2")
            big_endian = center_crop(native_stamps, 6)
            self.assertEqual(big_endian.dtype, native_stamps.dtype)
            np.testing.assert_array_equal(big_endian, old_crop(native_stamps, 6))

    def test_seeded_orientations_preserve_band_alignment_and_rng_state(self) -> None:
        random_state = np.random.get_state()
        self.addCleanup(np.random.set_state, random_state)
        stamp = np.arange(9 * 11 * 6, dtype="float32").reshape(9, 11, 6)
        mean = np.arange(6, dtype="f4")
        std = np.arange(1, 7, dtype="f4")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            base, extra = root / "photo.npz", root / "photo_wise.npz"
            np.savez(base, image=stamp[:, :, :4])
            np.savez(extra, image=stamp[:, :, 4:])
            ds = PhotozDataset(
                [base],
                [0.25],
                mean,
                std,
                augment=True,
                w1w2_dir=root,
                physics_augment=True,
            )
            expected = (stamp - mean) / (std + 1e-8)
            np.testing.assert_array_equal(
                load_npz(base, mean, std, extra)[0].permute(1, 2, 0).numpy(), expected
            )
            for seed in range(40):
                r = np.random.RandomState(seed)
                value = stamp
                z = 0.25
                if r.rand() < 0.4:
                    z += r.uniform(0.0, 0.5)
                    value = value * ((1.25 / (1.0 + z)) ** 4)
                value = (value - mean) / (std + 1e-8)
                if r.rand() > 0.5:
                    value = np.flip(value, axis=0)
                if r.rand() > 0.5:
                    value = np.flip(value, axis=1)
                value = np.rot90(value, int(r.randint(0, 4)))
                np.random.seed(seed)
                tensor, target = ds[0]
                np.testing.assert_array_equal(tensor.permute(1, 2, 0).numpy(), value)
                self.assertEqual(target.item(), np.float32(z))
                self.assertEqual(np.random.rand(), r.rand())

                r = np.random.RandomState(seed)
                draw = r.rand()
                value = stamp
                if draw < 0.5:
                    value = np.rot90(value, int(r.randint(0, 4)))
                elif draw < 0.8:
                    value = np.flip(value, axis=int(r.randint(0, 2)))
                np.random.seed(seed)
                np.testing.assert_array_equal(random_rot90_flip(stamp, 0.5, 0.3), value)
                self.assertEqual(np.random.rand(), r.rand())
