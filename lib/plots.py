from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray


def redshift_scatter(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
    path: str | Path,
    model_name: str | None,
    vmin: Any = None,
    vmax: Any = None
) -> None:
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=(6, 6))

    if vmin is None:
        vmin = np.min([y_true_arr.min(), y_pred_arr.min()])
    if vmax is None:
        vmax = np.max([y_true_arr.max(), y_pred_arr.max()])

    hb = ax.hexbin(
        y_true_arr,
        y_pred_arr,
        gridsize=120,
        cmap="plasma",
        mincnt=1
    )

    ax.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1)

    ax.plot(
        [vmin, vmax],
        [vmin + 0.15, vmax + 0.15],
        linestyle=":",
        linewidth=1
    )

    ax.plot(
        [vmin, vmax],
        [vmin - 0.15, vmax - 0.15],
        linestyle=":",
        linewidth=1
    )

    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Galaxies per pixel")

    ax.set_xlabel("z_spec")
    ax.set_ylabel("z_phot")
    ax.set_title(model_name if model_name is not None else "unknown")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
