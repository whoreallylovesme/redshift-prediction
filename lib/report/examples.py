from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from .task import Floats


def posterior_grid(method: dict[str, Any]) -> Floats | None:
    if not method.get("posterior_column"):
        return None
    grid = np.asarray(method.get("posterior_grid"), dtype=float)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all() or (np.diff(grid) <= 0).any():
        raise ValueError("Posterior requires a finite, strictly increasing grid of point masses")
    return grid


def probabilities(value: Any, grid: Floats) -> Floats | None:
    mass = np.asarray(value, dtype=float)
    if mass.shape != grid.shape or not np.isfinite(mass).all() or np.any(mass < 0):
        return None
    total = float(mass.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    return mass / total


def posterior_diagnostics(frame: pd.DataFrame, method: dict[str, Any]) -> dict[str, Any] | None:
    grid = posterior_grid(method)
    if grid is None:
        return None
    levels = np.array([0.5, 0.68, 0.8, 0.9, 0.95])
    totals = np.zeros(len(levels))
    widths = np.zeros(len(levels))
    used = 0
    for truth, value in zip(frame._truth, frame[method["posterior_column"]], strict=True):
        mass = probabilities(value, grid)
        if mass is None:
            continue
        cdf = np.cumsum(mass)
        cdf[-1] = 1.0
        low = grid[np.searchsorted(cdf, (1 - levels) / 2)]
        high = grid[np.searchsorted(cdf, (1 + levels) / 2)]
        totals += (low <= truth) & (truth <= high)
        widths += high - low
        used += 1
    return {
        "count": used,
        "invalid": len(frame) - used,
        "nominal": levels.tolist(),
        "empirical": (totals / used).tolist() if used else [],
        "mean_width": (widths / used).tolist() if used else [],
    }


def thumbnail(value: Any, source: Any) -> str | None:
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value)
    if not path.is_absolute() and isinstance(source, (str, Path)):
        path = Path(source).parent / path
    try:
        with Image.open(path) as img:
            if img.width * img.height > 16_000_000:
                return None
            img.thumbnail((144, 112))
            tiny = BytesIO()
            img.convert("RGB").save(tiny, format="JPEG", quality=76)
        return "data:image/jpeg;base64," + base64.b64encode(tiny.getvalue()).decode("ascii")
    except (ValueError, OSError, Image.DecompressionBombError):
        return None


def example_assets(row: pd.Series, method: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    image_col = method.get("thumbnail_column")
    if image_col:
        result["thumbnail"] = thumbnail(row.get(image_col), method.get("predictions"))
    grid = posterior_grid(method)
    if grid is not None:
        mass = probabilities(row.get(method["posterior_column"]), grid)
        if mass is not None:
            intervals = np.array_split(np.arange(len(grid)), min(len(grid), 160))
            result["posterior"] = {
                "x": [float(grid[idx].mean()) for idx in intervals],
                "y": [float(mass[idx].sum()) for idx in intervals],
                "raw_bins": len(grid),
            }
    return result


def native_posteriors(
    frame: pd.DataFrame,
    method: dict[str, Any],
    has_id: bool,
) -> list[dict[str, Any]]:
    if not method.get("posterior_column"):
        return []
    selected = frame.loc[frame._predicted]
    positions = np.linspace(0, len(selected) - 1, min(len(selected), 32), dtype=int)
    examples = []
    for position in positions:
        row = selected.iloc[position]
        assets = example_assets(row, method)
        if assets.get("posterior"):
            examples.append(
                {
                    "id": str(row.name) if has_id else None,
                    "prediction": float(row["_pred"]),
                    **assets,
                }
            )
        if len(examples) == 4:
            break
    return examples
