from pathlib import Path
import yaml
import pandas as pd
import click

from lib.dataset import clean_dataset

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--features", "features_path", required=True, type=click.Path(exists=True))
@click.option("--outdir", default="prepared")
def main(input_path, config_path, features_path, outdir):
    cfg = load_yaml(config_path)
    fcfg = load_yaml(features_path)

    df = pd.read_csv(input_path)
    df = clean_dataset(df, fcfg["target"], cfg["preprocess"])

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df.to_parquet(outdir / "data.parquet", index=False)

    meta = pd.DataFrame([{
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
    }])
    meta.to_parquet(outdir / "meta.parquet", index=False)

    print("Saved:", outdir / "data.parquet")
    print("Saved:", outdir / "meta.parquet")

if __name__ == "__main__":
    main()
