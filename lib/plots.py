from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def redshift_scatter(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
    path: str | Path,
) -> None:
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=(6, 6))

    vmin = np.min([y_true_arr.min(), y_pred_arr.min()])
    vmax = np.max([y_true_arr.max(), y_pred_arr.max()])

    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1)
    ax.scatter(y_true, y_pred, s=8, alpha=0.35)

    ax.set_xlabel("z_true")
    ax.set_ylabel("z_pred")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
