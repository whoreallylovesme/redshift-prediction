from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATHS = [
    "Teddy/teddy_A",
    "Teddy/teddy_B",
    "Teddy/teddy_C",
    "Teddy/teddy_D",
    "Happy/happy_A",
    "Happy/happy_B",
    "Happy/happy_C",
    "Happy/happy_D",
]


COLS = [
    "id", "mag_r", "u-g", "g-r", "r-i", "i-z",
    "z_spec", "feat1", "feat2", "feat3", "feat4", "feat5"
]


def load(path: str):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=COLS,
        engine="python",
    )

    X = df[["feat1", "feat2", "feat3", "feat4", "feat5"]].to_numpy(dtype=np.float64)
    y = df["z_spec"].to_numpy(dtype=np.float64)

    return X, y


def main():
    for input_path in INPUT_PATHS:
        input_path = Path(input_path)

        outdir = Path(input_path.parts[0] + "Parquet")
        outdir.mkdir(parents=True, exist_ok=True)

        X, y = load(str(input_path))

        df = pd.DataFrame(
            X,
            columns=["feat1", "feat2", "feat3", "feat4", "feat5"],
        )
        df["z_spec"] = y

        output_file = outdir / f"{input_path.name}.parquet"
        df.to_parquet(output_file, index=False, engine="pyarrow")

        print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
