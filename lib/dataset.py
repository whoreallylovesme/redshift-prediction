import pandas as pd
import numpy as np

def read_dataset(path):
    return pd.read_parquet(path)

def clean_dataset(df, target, preprocess_cfg):
    df = df[df[target].notna()]

    if preprocess_cfg["drop_star_spectype"]:
        if "DESIDR1_SPECTYPE" in df.columns:
            df = df[df["DESIDR1_SPECTYPE"] != "b'STAR  '"]

    if preprocess_cfg["drop_gaia_stars"]:
        if "ls_gaia_star_sn4" in df.columns:
            df = df[df["ls_gaia_star_sn4"] == False]

    if preprocess_cfg["drop_unnamed"]:
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

    df = df.replace([np.inf, -np.inf], np.nan)

    return df
