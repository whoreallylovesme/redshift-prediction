from pathlib import Path
from typing import Any

import click
import pandas as pd
import yaml

from lib.dataset import clean_dataset


def load_yaml(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--features", "features_path", required=True, type=click.Path(exists=True))
@click.option("--outdir", default="prepared")
def main(input_path: str, features_path: str, outdir: Path) -> None:
    fcfg = load_yaml(features_path)

    df = pd.read_csv(input_path, low_memory=False)

    df = clean_dataset(df, fcfg["target"])
    df = df.select_dtypes(include=["number", "bool"])

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(outdir / "data.parquet", index=False, engine="pyarrow")

    meta = pd.DataFrame([{
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }])
    meta.to_parquet(outdir / "meta.parquet", index=False, engine="pyarrow")

    print("Saved:", outdir / "data.parquet")
    print("Saved:", outdir / "meta.parquet")


if __name__ == "__main__":
    main()
