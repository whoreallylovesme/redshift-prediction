from __future__ import annotations

import numpy as np
import numpy.typing as npt


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top : top + size, left : left + size]


def subsample_indices(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))


def z_sample_weights(z: npt.NDArray[np.floating], n_bins: int = 20) -> npt.NDArray[np.float64]:
    bins = np.linspace(z.min(), z.max() + 1e-8, n_bins + 1)
    bin_idx = np.clip(np.digitize(z, bins) - 1, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    counts[counts == 0] = 1.0
    weights: npt.NDArray[np.float64] = (1.0 / counts)[bin_idx]
    return weights


def random_rot90_flip(
    img: np.ndarray, rot90_prob: float, flip_prob: float,
) -> np.ndarray:
    r = np.random.rand()
    if r < rot90_prob:
        k = np.random.randint(0, 4)
        if k > 0:
            return np.rot90(img, k).copy()
        return img
    if r < rot90_prob + flip_prob:
        axis = np.random.randint(0, 2)
        return np.flip(img, axis=axis).copy()
    return img
