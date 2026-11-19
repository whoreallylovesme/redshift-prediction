from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
import torch

from lib.dataloaders.common import d4
from lib.dataloaders.images import channel_stats, pixels
from lib.dataloaders.npz import read_npz_pixels
from photoz.data.npz import read_bands
from photoz.data.teddy_capsnet import TeddyCapsnetDataset


class SharedImageOperationsTest(TestCase):
    def test_d4_preserves_each_samples_channels_for_all_eight_orientations(self) -> None:
        image = torch.arange(8 * 3 * 25).reshape(8, 3, 5, 5)
        codes = torch.tensor([2, 5, 7, 0, 3, 1, 6, 4])
        for invalid in (float("nan"), 2.5, 8.0):
            invalid_codes = codes.float()
            invalid_codes[0] = invalid
            with self.subTest(code=invalid), self.assertRaises(ValueError):
                d4(image, invalid_codes)
        result = d4(image, codes)
        for i, code in enumerate(codes.tolist()):
            expected = np.rot90(image[i].numpy(), code % 4, axes=(1, 2))
            if code >= 4:
                expected = expected[:, :, ::-1]
            np.testing.assert_array_equal(result[i].numpy(), expected, strict=True)

    def test_channel_stats_weight_pixels_across_unequal_sizes_and_keep_zero_std(self) -> None:
        with TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{i}.npz" for i in (0, 1)]
            images = [np.full((2, 3, 2), 5), np.full((4, 4, 2), 5)]
            images[0][:, :, 0] = 12
            for path, image in zip(paths, images, strict=True):
                np.savez(path, sensor=image)
            mean, std = channel_stats(paths, "sensor")
            all_pixels = np.concatenate([image.reshape(-1, 2) for image in images])
            np.testing.assert_array_equal(mean, all_pixels.mean(axis=0).astype("f4"), strict=True)
            np.testing.assert_array_equal(std, all_pixels.std(axis=0).astype("f4"), strict=True)

    def test_npz_consumers_keep_their_validation_and_channel_selection(self) -> None:
        with TemporaryDirectory() as temp:
            path = Path(temp) / "003.npz"
            extra = Path(temp) / "003_wise.npz"
            base_pixels = np.arange(24, dtype=np.float64).reshape(2, 2, 6)
            base_pixels[:, :, 5] = np.nan
            separate = np.arange(12).reshape(2, 2, 3)
            np.savez(path, image=base_pixels, sensor=base_pixels[:, :, 0])
            np.savez(extra, image=separate)
            np.testing.assert_array_equal(
                read_npz_pixels(path, "sensor"), base_pixels[:, :, 0].astype("f4"), strict=True
            )
            self.assertEqual(pixels(path, "sensor").shape, (2, 2, 1))
            with self.assertRaisesRegex(ValueError, "finite"):
                pixels(path, "image")
            expected = np.concatenate([base_pixels[:, :, :4], separate[:, :, :2]], axis=2)
            np.testing.assert_array_equal(read_bands(path, 4, extra), expected.astype("f4"))
            dataset = TeddyCapsnetDataset(
                [path], np.array([0.1]), np.array([1]), in_channels=6, w1w2_dir=extra.parent,
            )
            image, _, _ = dataset[0]
            np.testing.assert_array_equal(
                image.permute(1, 2, 0).numpy(), expected.astype("f4") / 10, strict=True
            )
