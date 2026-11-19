import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype


def exact_ids(frame: pd.DataFrame, columns: tuple[str, ...] = ("object_id", "tile")) -> None:
    for name in columns:
        values = frame[name]
        if (
            not is_integer_dtype(values.dtype)
            or values.isna().any()
            or not values.between(0, np.iinfo(np.int64).max).all()
        ):
            raise ValueError(f"{name} requires exact nonnegative signed-64-bit identities")
