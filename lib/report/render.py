from __future__ import annotations

import csv
import json
import math
from importlib.resources import files
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment
from plotly.offline import get_plotlyjs

from .evaluation import Record, build_report_data
from .task import ReportTask


def json_value(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def replace_text(path: Path, text: str) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(text)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)


def export_records(data: Record) -> Record:
    out: Record = {k: data[k] for k in ("title", "task", "primary", "metrics", "overlaps")}
    out["rows"] = [
        {
            **{k: row[k] for k in ("name", "split", "role", "status", "metrics", "coverage")},
            "reason": row.get("details", {}).get("reason"),
            "slices": {
                name: {"coverage": value["coverage"], "metrics": value["metrics"]}
                for name, value in row["slices"].items()
            },
        }
        for row in data["rows"]
    ]
    out["pairs"] = [
        {
            **p,
            "slices": {
                name: {k: v for k, v in val.items() if k != "examples"}
                for name, val in p["slices"].items()
            },
        }
        for p in data["pairs"]
    ]
    return out


def write_report(outdir: Path, report_data: Record) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data = json_value(report_data)
    payload = json.dumps(data, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
    template = Environment(autoescape=True).from_string(
        files("lib.report").joinpath("view.html").read_text(encoding="utf-8")
    )
    html = template.render(
        title=data["title"], payload=payload.replace("<", "\\u003c"), plotly=get_plotlyjs()
    )
    replace_text(outdir / "report-data.json", payload + "\n")
    export = export_records(data)
    replace_text(outdir / "metrics.json", json.dumps(export, ensure_ascii=False, indent=2) + "\n")
    columns = [
        "name",
        "split",
        "role",
        "status",
        "slice",
        "requested",
        "admitted",
        "predicted",
        "evaluated",
        *data["metrics"],
    ]
    with NamedTemporaryFile("w", encoding="utf-8", dir=outdir, newline="", delete=False) as stream:
        tmp = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in export["rows"]:
                base = {k: row[k] for k in ("name", "split", "role", "status")}
                for name, val in row["slices"].items():
                    coverage = {k: v for k, v in val["coverage"].items() if k != "reasons"}
                    writer.writerow({**base, "slice": name, **coverage, **val["metrics"]})
                if not row["slices"]:
                    writer.writerow(base)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    tmp.replace(outdir / "metrics.csv")
    replace_text(outdir / "report.html", html)


def render_report(
    outdir: Path,
    methods: list[Record],
    task: str | ReportTask,
    title: str,
    metadata: Record | None = None,
) -> Record:
    data = build_report_data(methods, task, title, metadata)
    write_report(outdir, data)
    return data
