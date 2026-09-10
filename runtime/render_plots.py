#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from lib.dataset import read_dataset
from lib.plots import redshift_scatter

TEMPLATES_DIR = Path(__file__).parent / "templates"


@dataclass(frozen=True)
class Tile:
    algo: str
    img_path: Path


def clean_algo_name(name: str) -> str:
    return re.sub(r"_set\d+$", "", name)


def img_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def load_tiles(runs_dir: Path) -> list[Tile]:
    tiles: list[Tile] = []

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        plots_dir = run_dir / "plots"
        if not plots_dir.is_dir():
            continue

        algo = clean_algo_name(run_dir.name)

        for p in sorted(plots_dir.glob("*.png")):
            if p.is_file():
                tiles.append(Tile(algo=algo, img_path=p))

    return tiles


def generate_scatter_plots(runs_dir: str) -> None:
    runs_path = Path(runs_dir)
    for run_dir in sorted(p for p in runs_path.iterdir() if p.is_dir()):
        preds_path = run_dir / "preds.parquet"
        report_path = run_dir / "report.json"
        if not preds_path.exists() or not report_path.exists():
            continue

        plots_dir = run_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        report = json.loads(report_path.read_text(encoding="utf-8"))
        model_name = report.get("model_name", run_dir.name)

        df = read_dataset(str(preds_path))
        y_true = df["y_true"]
        y_pred = df["y_pred"].to_numpy()

        if model_name == "llr":
            redshift_scatter(y_true, y_pred, plots_dir / "scatter.png", model_name, 0, 4)
        else:
            redshift_scatter(y_true, y_pred, plots_dir / "scatter.png", model_name)


def summary_plots(runs_dir: str, out: str, title: str = "main") -> None:
    generate_scatter_plots(runs_dir)

    runs_path = Path(runs_dir)
    out_path = Path(out)

    tiles = load_tiles(runs_path)
    if not tiles:
        return

    tiles_data = [
        {"algo": t.algo, "uri": img_data_uri(t.img_path)} for t in tiles
    ]

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"]),
    )
    html_text = env.get_template("summary_plots.html.j2").render(
        title=title, tiles=tiles_data,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
