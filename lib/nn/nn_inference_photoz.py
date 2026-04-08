from pathlib import Path

import numpy as np
import torch


def load_npz(
    path: Path, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:

    img = np.load(path)["image"].astype(np.float32)[:, :, :4]
    img = (img - mean) / (std + 1e-8)
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
