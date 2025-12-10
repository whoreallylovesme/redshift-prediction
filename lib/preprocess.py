import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from features import get_features_set

TARGET = "DESIDR1_SPECTYPE"
# TARGET_FALLBACK = "lg(LX2-10)_sp"


def preprocess(
    df,
    feature_set_num=1,
    test_size=0.2,
    val_size=0.25,
    n_strata=5,
):
    df = df[(df[TARGET] != "b'STAR  '") & (df[TARGET].notna())]

    if "ls_gaia_star_sn4" in df.columns:
        df = df[df["ls_gaia_star_sn4"] == False]

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])


    features = get_features_set(df, feature_set_num)


    X = df[features]
    y = df[TARGET]

    X = X.select_dtypes(include=["float64", "float32", "int64", "int32"]).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    stratify_labels = None
    if n_strata is not None:
        try:
            stratify_labels = pd.qcut(y, q=n_strata, labels=False, duplicates="drop")
        except ValueError:
            stratify_labels = None

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify_labels,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_full,
        y_train_full,
        features,
    )
