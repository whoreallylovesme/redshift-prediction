import pandas as pd

from lib.dataloaders.dataset import clean_regression_data


def clean_dataset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if "DESIDR1_SPECTYPE" in df.columns:
        df = df[df["DESIDR1_SPECTYPE"] != "b'STAR  '"]
    if "ls_gaia_star_sn4" in df.columns:
        df = df[~df["ls_gaia_star_sn4"]]
    return clean_regression_data(df, target)
