from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from lib.config import DatasetConfig, FeaturesConfig
from lib.utils.utils import load_yaml_as
from photoz.data.tabular import clean_dataset


def prepare(
    input_csv: str, data_config: str, features: str, outdir: str, id_column: str | None = None
) -> None:
    dcfg = load_yaml_as(data_config, DatasetConfig)
    fcfg = load_yaml_as(features, FeaturesConfig)
    target = fcfg.target

    if id_column is not None and id_column in {target, "y_true", "y_pred"}:
        raise ValueError("ID and target columns must be distinct")
    df = pd.read_csv(
        input_csv, low_memory=False, dtype={id_column: "string"} if id_column else None
    )
    df = clean_dataset(df, target)
    numeric = df.select_dtypes(include=["number", "bool"]).copy()
    if id_column is not None:
        ids = df[id_column]
        if ids.isna().any() or not ids.is_unique:
            raise ValueError("Legacy dataset requires unique non-null IDs")
        numeric[id_column] = ids
    df = numeric.reset_index(drop=True)

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

    meta = pd.DataFrame(
        {
            "input": [str(Path(input_csv).resolve())],
            "target": [target],
            "rows_total": [len(df)],
            "rows_train": [len(train_df)],
            "rows_test": [len(test_df)],
            "cols": [df.shape[1]],
            "split_test_size": [dcfg.test_size],
            "split_shuffle": [dcfg.shuffle],
            "split_random_seed": [dcfg.random_seed],
        }
    )
    meta.to_parquet(out / "meta.parquet", index=False, engine="pyarrow")
