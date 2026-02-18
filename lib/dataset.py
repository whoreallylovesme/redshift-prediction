import numpy as np
import pandas as pd


def read_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow", use_threads=False)


def clean_dataset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df[df[target].notna()]

    if "DESIDR1_SPECTYPE" in df.columns:
        df = df[df["DESIDR1_SPECTYPE"] != "b'STAR  '"]

    if "ls_gaia_star_sn4" in df.columns:
        df = df[~df["ls_gaia_star_sn4"]]

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.replace([np.inf, -np.inf], np.nan)

    return df
