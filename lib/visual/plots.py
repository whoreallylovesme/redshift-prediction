import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray


def redshift_scatter(
    y_true: pd.Series,
    y_pred: NDArray[np.float64],
    model_name: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> go.Figure:
    y_true_arr = np.asarray(y_true).ravel()
    y_pred_arr = np.asarray(y_pred).ravel()

    if vmin is None:
        vmin = float(np.min([y_true_arr.min(), y_pred_arr.min()]))
    if vmax is None:
        vmax = float(np.max([y_true_arr.max(), y_pred_arr.max()]))

    fig = go.Figure()

    fig.add_trace(go.Histogram2d(
        x=y_true_arr,
        y=y_pred_arr,
        nbinsx=120,
        nbinsy=120,
        colorscale="Plasma",
        zmin=1,
        hovertemplate="z_spec=%{x:.4f}<br>z_phot=%{y:.4f}<br>count=%{z}<extra></extra>",
    ))

    for offset, dash, name in [
        (0.0, "dash", "z_phot = z_spec"),
        (0.15, "dot", "+0.15"),
        (-0.15, "dot", "-0.15"),
    ]:
        fig.add_trace(go.Scatter(
            x=[vmin, vmax], y=[vmin + offset, vmax + offset],
            mode="lines", line={"dash": dash, "width": 1, "color": "gray"},
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
