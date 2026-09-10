from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
PITT = REPO / "datasets" / "dey2022" / "pitt"

TARGETS = [
    ("datasets/sdss_w_specz_train.h5", "datasets/dey2022/hayat_train_gz1.parquet"),
    ("sdss_w_specz_valid.h5", "datasets/dey2022/hayat_valid_gz1_pitt.parquet"),
]


def build_label_map() -> dict[int, float]:
    label_map: dict[int, float] = {}
    for split in ["train", "dev", "test"]:
        cat = np.load(PITT / f"42023_cat_{split}.npy", allow_pickle=True)
        objids = np.asarray(cat["objID"].tolist(), dtype=np.int64)
        for oid, gz in zip(objids.tolist(), cat["gz1Morpho"].tolist(), strict=True):
            if not np.isnan(gz):
                label_map[oid] = float(gz)
    print(f"[join] Pitt confident labels: {len(label_map)} galaxies")
    return label_map


def main() -> None:
    label_map = build_label_map()
    for h5_name, out_name in TARGETS:
        h5_path = REPO / h5_name
        if not h5_path.exists():
            print(f"[join] SKIP (missing): {h5_name}")
            continue
        with h5py.File(h5_path, "r") as f:
            objids = [int(x) for x in f["ObjID"][:].astype(str)]
        labels = np.array([label_map.get(o, np.nan) for o in objids], dtype=np.float32)
        out = pd.DataFrame({"objid": objids, "gz_morpho_binary": labels})
        out_path = REPO / out_name
        out.to_parquet(out_path, index=False)
        n = int((~np.isnan(labels)).sum())
        print(
            f"[join] {out_name}: N={len(out)}, labeled={n} ({100 * n / len(out):.1f}%), "
            f"spiral={int((labels == 1).sum())}, ellip={int((labels == 0).sum())}"
        )


if __name__ == "__main__":
    main()
