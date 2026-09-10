from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lib.config_models import DatasetConfig, FeaturesConfig
from lib.dataset import clean_dataset
from lib.utils import load_yaml_as


def prepare(input_csv: str, data_config: str, features: str, outdir: str) -> None:
    dcfg = load_yaml_as(data_config, DatasetConfig)
    fcfg = load_yaml_as(features, FeaturesConfig)
    target = fcfg.target

    df = pd.read_csv(input_csv, low_memory=False)
    df = clean_dataset(df, target)
    df = df.select_dtypes(include=["number", "bool"]).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=dcfg.test_size,
        random_state=dcfg.random_seed,
        shuffle=dcfg.shuffle,
    )

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(out / "train.parquet", index=False, engine="pyarrow")
    test_df.to_parquet(out / "test.parquet", index=False, engine="pyarrow")

    meta = pd.DataFrame({
            "input": [str(Path(input_csv).resolve())],
            "target": [target],
            "rows_total": [len(df)],
            "rows_train": [len(train_df)],
            "rows_test": [len(test_df)],
            "cols": [df.shape[1]],
            "split_test_size": [dcfg.test_size],
            "split_shuffle": [dcfg.shuffle],
            "split_random_seed": [dcfg.random_seed],
    })
    meta.to_parquet(out / "meta.parquet", index=False, engine="pyarrow")
