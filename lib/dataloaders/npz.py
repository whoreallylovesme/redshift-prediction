from pathlib import Path

import numpy as np
import numpy.typing as npt


def read_npz_pixels(path: Path, key: str) -> npt.NDArray[np.float32]:
    with np.load(path, allow_pickle=False) as archive:
        return np.asarray(archive[key], dtype=np.float32)
