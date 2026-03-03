#!/usr/bin/env python3
from __future__ import annotations

import base64
import html
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path

import click


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


def render_html(tiles: list[Tile], title: str) -> str:
    cards: list[str] = []

    for t in tiles:
        uri = img_data_uri(t.img_path)
        cards.append(
            "<div class='card'>"
            f"<img src='{html.escape(uri)}' alt='{html.escape(t.algo)}'/>"
            f"<div class='cap'>{html.escape(t.algo)}</div>"
            "</div>"
        )

    content = "\n".join(cards)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{html.escape(title)}</title>
<style>
body {{
  font-family: system-ui;
  margin: 24px;
}}

h1 {{
  font-size: 18px;
  margin-bottom: 16px;
}}

.grid {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 14px;
}}

@media (max-width: 1100px) {{
  .grid {{ grid-template-columns: repeat(2, 1fr); }}
}}
@media (max-width: 700px) {{
  .grid {{ grid-template-columns: 1fr; }}
}}

.card {{
  border: 1px solid #eee;
  border-radius: 12px;
  padding: 10px;
}}

img {{
  width: 100%;
  height: auto;
  border-radius: 10px;
  border: 1px solid #eee;
}}

.cap {{
  margin-top: 8px;
  font-size: 13px;
  font-weight: 700;
  text-align: center;
}}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="grid">
{content}
</div>
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
    default=Path("runs/plots_gallery.html"),
    show_default=True,
)
@click.option("--title", type=str, default="main", show_default=True)
def main(runs_dir: Path, out: Path, title: str) -> None:
    tiles = load_tiles(runs_dir)

    if not tiles:
        print(f"No runs/*/plots/*.png found in {runs_dir}")
        return

    html_text = render_html(tiles, title)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_text, encoding="utf-8")
    print(f"Saved plots gallery to {out}")


if __name__ == "__main__":
    main()
