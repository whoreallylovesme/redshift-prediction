import pandas as pd

from sklearn.model_selection import train_test_split

TARGET = "DESIDR1_SPECTYPE"
# TARGET_FALLBACK = "lg(LX2-10)_sp"

def preprocess(df):
    df = df[
        (df[TARGET] != "b'STAR  '") &
        (df[TARGET].notna())
    ]

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    features = config["features"]
    X = df[features]
    y = df[TARGET]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=pd.qcut(y, q=5, labels=False)
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    df = df.select_dtypes(include=["float64"])
    df = df.replace([np.inf, -np.inf], np.nan)
