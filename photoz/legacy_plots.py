import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray

from photoz.report import PHOTOZ


def redshift_scatter(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
    model_name: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> go.Figure:
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    valid = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr, y_pred_arr = y_true_arr[valid], y_pred_arr[valid]
    if vmin is None:
        vmin = float(min(y_true_arr.min(), y_pred_arr.min())) if valid.any() else 0.0
    if vmax is None:
        vmax = float(max(y_true_arr.max(), y_pred_arr.max())) if valid.any() else 1.0
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    fig = go.Figure()
    if not valid.any():
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            text="Нет оценённых объектов",
        )

    fig.add_trace(go.Histogram2d(
        x=y_true_arr,
        y=y_pred_arr,
        nbinsx=120,
        nbinsy=120,
        colorscale="Plasma",
        zmin=1,
        hovertemplate="z_spec=%{x:.4f}<br>z_phot=%{y:.4f}<br>count=%{z}<extra></extra>",
    ))

    bounds = np.array([vmin, vmax])
    for i, (name, values) in enumerate(PHOTOZ.guides(bounds).items()):
        fig.add_trace(go.Scatter(
            x=bounds, y=values, mode="lines",
            line={"dash": "dash" if i == 0 else "dot", "width": 1, "color": "gray"},
            name=name, hoverinfo="skip",
        ))

    fig.update_layout(
        title=model_name if model_name is not None else "",
        xaxis_title="z_spec",
        yaxis_title="z_phot",
        xaxis_range=[vmin, vmax],
        yaxis_range=[vmin, vmax],
        width=600,
        height=600,
        coloraxis_colorbar={"title": "Count"},
        template="plotly_white",
    )
    return fig
