#!/usr/bin/env python3

import json
from pathlib import Path

import click
import pandas as pd

TEDDY_FILES = [
    "useful_info/teddy_happy/Teddy/teddy_A",
    "useful_info/teddy_happy/Teddy/teddy_B",
    "useful_info/teddy_happy/Teddy/teddy_C",
    "useful_info/teddy_happy/Teddy/teddy_D",
]

HAPPY_FILES = [
    "useful_info/teddy_happy/Happy/happy_A",
    "useful_info/teddy_happy/Happy/happy_B",
    "useful_info/teddy_happy/Happy/happy_C",
    "useful_info/teddy_happy/Happy/happy_D",
]

COLS = [
    "id", "mag_r", "u-g", "g-r", "r-i", "i-z",
    "z_spec", "feat1", "feat2", "feat3", "feat4", "feat5",
]


def load_ids(path: str) -> pd.Series:
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=COLS,
        engine="python",
    )
    return df["id"]


def collect_ids(paths: list[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for path in paths:
        split_name = path.split("_")[-1].upper()
        result[split_name] = load_ids(path).astype("int64").tolist()
    return result


@click.command()
@click.option(
    "--outdir",
    default=".",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
def main(outdir: Path) -> None:
    teddy_ids = collect_ids(TEDDY_FILES)
    happy_ids = collect_ids(HAPPY_FILES)

    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "teddy_ids.json", "w", encoding="utf-8") as f:
        json.dump(teddy_ids, f, ensure_ascii=False)

    with open(outdir / "happy_ids.json", "w", encoding="utf-8") as f:
        json.dump(happy_ids, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
