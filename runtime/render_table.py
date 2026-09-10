#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).parent / "templates"

LOWER_IS_BETTER: set[str] = {"rmse", "sigma_nmad", "f_outlier_percent"}
CLOSER_TO_ZERO_IS_BETTER: set[str] = {"bias"}

DISPLAY_SCALE: dict[str, float] = {"bias": 1e2}
DISPLAY_SUFFIX: dict[str, str] = {"bias": "10^-2"}


@dataclass(frozen=True)
class Row:
    algo: str
    test: dict[str, float]


def clean_algo_name(name: str) -> str:
    return re.sub(r"_set\d+$", "", name)


def get_test_dict(data: dict[str, Any]) -> dict[str, Any]:
    x = data.get("test_metrics")
    if isinstance(x, dict):
        return x
    x = data.get("test")
    if isinstance(x, dict):
        return x
    m = data.get("metrics")
    if isinstance(m, dict):
        x = m.get("test")
        if isinstance(x, dict):
            return x
    return {}


def to_float_dict(d: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def load_rows(runs_dir: Path) -> list[Row]:
    rows: list[Row] = []
    for report_path in sorted(runs_dir.glob("*/report.json")):
        run_dir = report_path.parent
        data: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))

        algo = clean_algo_name(run_dir.name)
        test = to_float_dict(get_test_dict(data))

        rows.append(Row(algo=algo, test=test))

    return rows


def all_metrics(rows: list[Row]) -> list[str]:
    keys: set[str] = set()
    for r in rows:
        keys |= set(r.test.keys())

    ordered = [k for k in ["rmse", "sigma_nmad", "f_outlier_percent", "bias"] if k in keys]
    return ordered


def best_map(rows: list[Row], metrics: list[str]) -> dict[str, float]:
    best: dict[str, float] = {}
    for m in metrics:
        vals: list[float] = [r.test[m] for r in rows if m in r.test]
        if not vals:
            continue
        if m in CLOSER_TO_ZERO_IS_BETTER:
            best[m] = min(abs(v) for v in vals)
        else:
            best[m] = min(vals)
    return best


def is_best(value: float, best_value: float, metric: str) -> bool:
    eps = 1e-12
    if metric in CLOSER_TO_ZERO_IS_BETTER:
        return abs(value) <= best_value + eps
    return value <= best_value + eps


def fmt(x: Any) -> str:
    if isinstance(x, (int, float)):
        return f"{x:.3g}"
    return str(x)


def _build_cells(
    r: Row, metrics: list[str], best: dict[str, float],
) -> list[dict[str, str]]:
    cells: list[dict[str, str]] = []
    for m in metrics:
        if m not in r.test:
            cells.append({"class": "num muted", "value": "—"})
            continue
        v = r.test[m]
        v_disp = v * DISPLAY_SCALE.get(m, 1.0)
        cls = "num"
        bv = best.get(m)
        if bv is not None and is_best(v, bv, m):
            cls += " best"
        cells.append({"class": cls, "value": fmt(v_disp)})
    return cells


def summary_table(runs_dir: str, out: str, title: str = "main") -> None:
    runs_path = Path(runs_dir)
    out_path = Path(out)

    rows = load_rows(runs_path)
    if not rows:
        return

    metrics = all_metrics(rows)
    best = best_map(rows, metrics)
    metric_labels = [
        {"key": m, "label": f"{m} ({DISPLAY_SUFFIX[m]})" if m in DISPLAY_SUFFIX else m}
        for m in metrics
    ]
    table_rows = [
        {"algo": r.algo, "cells": _build_cells(r, metrics, best)} for r in rows
    ]

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    html_text = env.get_template("summary_table.html.j2").render(
        title=title, metrics=metric_labels, rows=table_rows,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
