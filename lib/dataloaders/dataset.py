import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype


def clean_regression_data(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df.loc[df[target].notna()].drop(columns=["Unnamed: 0"], errors="ignore").replace(
        [np.inf, -np.inf], np.nan,
    )


def exact_index(values: pd.Series, name: str) -> pd.Index:
    ids = pd.Index(values)
    if ids.hasnans or not ids.is_unique:
        raise ValueError(f"{name} must contain unique, non-null IDs")
    if len(ids) and infer_dtype(ids, skipna=False) not in {"integer", "string"}:
        raise ValueError(f"{name} must contain exact integer or string IDs")
    return ids


def matching_id_types(left: pd.Index, right: pd.Index) -> bool:
    return not len(left) or not len(right) or infer_dtype(left) == infer_dtype(right)
