#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from lib.dataloaders.dataset import read_dataset
from lib.visual.plots import redshift_scatter

TEMPLATES_DIR = Path(__file__).parent / "templates"


def clean_algo_name(name: str) -> str:
    return re.sub(r"_set\d+$", "", name)


def load_charts(runs_dir: Path) -> list[dict[str, str]]:
    charts: list[dict[str, str]] = []

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        preds_path = run_dir / "preds.parquet"
        report_path = run_dir / "report.json"
        if not preds_path.exists() or not report_path.exists():
            continue

        algo = clean_algo_name(run_dir.name)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        model_name = report.get("model_name", run_dir.name)

        df = read_dataset(str(preds_path))
        y_true = df["y_true"]
        y_pred = df["y_pred"].to_numpy()

        if model_name == "llr":
            fig = redshift_scatter(y_true, y_pred, model_name, 0, 4)
        else:
            fig = redshift_scatter(y_true, y_pred, model_name)

        div = fig.to_html(
            full_html=False, include_plotlyjs=False,
            config={"displayModeBar": True, "responsive": True},
        )
        charts.append({"algo": algo, "div": div})

    return charts


def summary_plots(runs_dir: str, out: str, title: str = "main") -> None:
    runs_path = Path(runs_dir)
    out_path = Path(out)

    charts = load_charts(runs_path)
    if not charts:
        return

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    html_text = env.get_template("summary_plots.html.j2").render(
        title=title, charts=charts,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
