import numpy as np
import matplotlib.pyplot as plt


def redshift_scatter(y_true, y_pred, path):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=(6, 6))

    vmin = np.min([y_true.min(), y_pred.min()])
    vmax = np.max([y_true.max(), y_pred.max()])

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
