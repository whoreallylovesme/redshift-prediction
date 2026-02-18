from pathlib import Path
from typing import Any

import click
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from lib.dataset import clean_dataset


def load_yaml(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@click.command()
@click.option("--input", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--data-config", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--features", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--outdir", required=True, type=click.Path(file_okay=False))
def main(input: str, data_config: str, features: str, outdir: str) -> None:
    dcfg = load_yaml(data_config)
    split = dcfg["split"]

    test_size = float(split["test_size"])
    shuffle = bool(split["shuffle"])
    random_seed = int(split["random_seed"])

    target = load_yaml(features)["target"]

    df = pd.read_csv(input, low_memory=False)
    df = clean_dataset(df, target)
    df = df.select_dtypes(include=["number", "bool"]).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_seed,
        shuffle=shuffle,
    )

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out / "train.parquet", index=False, engine="pyarrow")
    test_df.to_parquet(out / "test.parquet", index=False, engine="pyarrow")

    meta = pd.DataFrame({
            "input": [str(Path(input).resolve())],
            "target": [target],
            "rows_total": [len(df)],
            "rows_train": [len(train_df)],
            "rows_test": [len(test_df)],
            "cols": [df.shape[1]],
            "split_test_size": [test_size],
            "split_shuffle": [shuffle],
            "split_random_seed": [random_seed],
    })
    meta.to_parquet(out / "meta.parquet", index=False, engine="pyarrow")

    print(f"Saved to {out.resolve()}")


if __name__ == "__main__":
    main()
