from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch


def load_npz(
    path: Path,
    mean: npt.NDArray[np.floating],
    std: npt.NDArray[np.floating],
    w1w2_path: Path | None = None,
    in_channels: int = 4,
) -> torch.Tensor:
    img = np.load(path)["image"].astype(np.float32)[:, :, :in_channels]
    if w1w2_path is not None:
        w1w2 = np.load(w1w2_path)["image"].astype(np.float32)[:, :, :2]
        img = np.concatenate([img, w1w2], axis=2)
    img = (img - mean) / (std + 1e-8)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
