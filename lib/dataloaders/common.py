from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as vision


def _spatial_tensor(img: np.ndarray) -> torch.Tensor:
    array = img.astype(img.dtype.newbyteorder("="), copy=False)
    if any(s < 0 for s in array.strides):
        array = array.copy()
    tensor = torch.from_numpy(array)
    return tensor.permute(2, 0, 1) if img.ndim == 3 else tensor


def center_crop(img: np.ndarray, size: int, *, out_size: int | None = None) -> np.ndarray:
    """Crop HWC or HW inputs with the historical floor offset, including odd margins."""
    h, w = img.shape[:2]
    top, left = (h - size) // 2, (w - size) // 2
    # Keep NumPy's slice bounds for legacy rectangular stamps smaller than the requested crop.
    top, bottom, _ = slice(top, top + size).indices(h)
    left, right, _ = slice(left, left + size).indices(w)
    image = _spatial_tensor(img)
    crop = (top, left, bottom - top, right - left)
    if out_size is not None and out_size != size:
        channels = image if img.ndim == 3 else image.unsqueeze(0)
        resized: torch.Tensor = vision.resized_crop(
            channels.float(), *crop, [out_size, out_size],
            interpolation=InterpolationMode.BILINEAR, antialias=False,
        )
        return (resized.permute(1, 2, 0) if img.ndim == 3 else resized[0]).numpy()
    cropped: torch.Tensor = vision.crop(image, *crop)
    output = cropped.permute(1, 2, 0) if img.ndim == 3 else cropped
    return output.numpy().astype(img.dtype, copy=False)


def resize_square(image: torch.Tensor, size: int) -> torch.Tensor:
    resized: torch.Tensor = vision.resize(
        image, [size, size], interpolation=InterpolationMode.BILINEAR, antialias=False,
    )
    return resized


def orient(
    image: torch.Tensor, *, vertical: bool = False, horizontal: bool = False, turns: int = 0,
) -> torch.Tensor:
    if vertical:
        image = vision.vertical_flip(image)
    if horizontal:
        image = vision.horizontal_flip(image)
    return vision.rotate(image, 90 * turns, expand=True) if turns else image


def subsample_indices(n: int, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=size, replace=False))


def inverse_bin_weights(
    target: npt.NDArray[np.floating], n_bins: int = 20
) -> npt.NDArray[np.float64]:
    bins = np.linspace(target.min(), target.max() + 1e-8, n_bins + 1)
    bin_idx = np.clip(np.digitize(target, bins) - 1, 0, n_bins - 1)
    counts = np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    counts[counts == 0] = 1.0
    weights: npt.NDArray[np.float64] = (1.0 / counts)[bin_idx]
    return weights


def z_sample_weights(z: npt.NDArray[np.floating], n_bins: int = 20) -> npt.NDArray[np.float64]:
    return inverse_bin_weights(z, n_bins)


def random_rot90_flip(
    img: np.ndarray, rot90_prob: float, flip_prob: float,
) -> np.ndarray:
    r = np.random.rand()
    turns, vertical, horizontal = 0, False, False
    if r < rot90_prob:
        turns = int(np.random.randint(0, 4))
    elif r < rot90_prob + flip_prob:
        vertical = np.random.randint(0, 2) == 0
        horizontal = not vertical
    if not (turns or vertical or horizontal):
        return img
    fixed = orient(_spatial_tensor(img), turns=turns, vertical=vertical, horizontal=horizontal)
    output = fixed.permute(1, 2, 0) if img.ndim == 3 else fixed
    return output.numpy().astype(img.dtype, copy=False)
