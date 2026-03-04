#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

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


def render_table(rows: list[Row], metrics: list[str], best: dict[str, float], title: str) -> str:
    parts: list[str] = []
    col_count = 1 + len(metrics)

    parts.append("<div class='table-wrap'><table>")

    parts.append("<thead>")
    parts.append(
        f"<tr><th class='table-title' colspan='{col_count}'>{html.escape(title)}</th></tr>"
    )

    parts.append("<tr>")
    parts.append("<th>algo</th>")

    for m in metrics:
        label = m
        if m in DISPLAY_SUFFIX:
            label = f"{m} ({DISPLAY_SUFFIX[m]})"
        parts.append(f"<th>{html.escape(label)}</th>")

    parts.append("</tr>")
    parts.append("</thead><tbody>")

    for r in rows:
        parts.append("<tr>")
        parts.append(f"<td>{html.escape(r.algo)}</td>")

        for m in metrics:
            if m not in r.test:
                parts.append("<td class='num muted'>—</td>")
                continue

            v = r.test[m]
            v_disp = v * DISPLAY_SCALE.get(m, 1.0)
            cls = "num"

            bv = best.get(m)
            if bv is not None and is_best(v, bv, m):
                cls += " best"

            parts.append(f"<td class='{cls}'>{html.escape(fmt(v_disp))}</td>")

        parts.append("</tr>")

    parts.append("</tbody></table></div>")
    return "\n".join(parts)


def render_html(rows: list[Row], title: str) -> str:
    metrics = all_metrics(rows)
    best = best_map(rows, metrics)
    content = render_table(rows, metrics, best, title)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
body {{ font-family: system-ui; margin: 24px; }}

table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #eee; padding: 6px 10px; }}
th {{ background: #fafafa; }}

.table-title {{
  font-size: 18px;
  font-weight: bold;
  text-align: center;
  background: #fafafa;
  padding: 8px 0;
}}

.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.best {{ font-weight: 800; background: #fff7cc; }}
.muted {{ color: #999; }}
</style>
</head>
<body>
{content}
</body>
</html>
"""


@click.command()
@click.option(
    "--runs-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("runs"),
    show_default=True,
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("runs/summary.html"),
    show_default=True,
)
@click.option(
    "--title",
    type=str,
    default="main",
    show_default=True,
)
def main(runs_dir: Path, out: Path, title: str) -> None:
    rows = load_rows(runs_dir)
    if not rows:
        print(f"No report.json found in {runs_dir}")
        return

    html_text = render_html(rows, title)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")

    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
