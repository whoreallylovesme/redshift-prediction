import numpy as np
import pandas as pd


COLS = [
    "id", "mag_r", "u-g", "g-r", "r-i", "i-z",
    "z_spec", "feat1", "feat2", "feat3", "feat4", "feat5"
]


def load(path: str):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=COLS,
        engine="python",
    )

    X = df[["feat1", "feat2", "feat3", "feat4", "feat5"]].to_numpy(dtype=np.float64)
    y = df["z_spec"].to_numpy(dtype=np.float64)

    return X, y