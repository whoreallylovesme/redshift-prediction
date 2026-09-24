#!/usr/bin/env python3
"""Нарезка 64x64x4 штампов из мозаик тайла (VIS,Y,J,H на одной сетке).

Аргументы: tile_id objects.parquet(колонки object_id, right_ascension, declination)
Выход: /root/euclid_q1/stamps/{tile}.npz (X float16 [N,4,64,64], ids int64,
ra, dec; каналы: VIS, Y, J, H).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

TILE = sys.argv[1]
OBJ = Path(sys.argv[2])
MOSAICS = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"/root/euclid_q1/mosaics/{TILE}")
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(f"/root/euclid_q1/stamps/{TILE}.npz")
CUT = 32  # полукадр 64x64

BANDS = ["vis", "y", "j", "h"]


def main() -> None:
    df = pd.read_parquet(OBJ)
    ra = df["right_ascension"].to_numpy(float)
    dec = df["declination"].to_numpy(float)
    ids = df["object_id"].to_numpy(np.int64)
    n = len(df)

    shape = None
    for band in BANDS:
        f = MOSAICS / f"{band}.fits"
        with fits.open(f, memmap=True) as hd:
            hdr = hd[0].header if hd[0].data is not None else hd[1].header
            data = hd[0].data if hd[0].data is not None else hd[1].data
            w = WCS(hdr).celestial
            x, y = w.world_to_pixel_values(ra, dec)
            xi = np.rint(x).astype(int)
            yi = np.rint(y).astype(int)
            if shape is None:
                shape = data.shape
                H, W = shape[-2], shape[-1]
                inside = (xi >= CUT) & (xi < W - CUT) & (yi >= CUT) & (yi < H - CUT)
                print(f"{TILE}: объектов {n}, внутри кадра {inside.sum()}")
                xi, yi = xi[inside], yi[inside]
                ids_in = ids[inside]
                ra_in, dec_in = ra[inside], dec[inside]
                X = np.zeros((inside.sum(), 4, 2 * CUT, 2 * CUT), dtype=np.float32)
            else:
                # та же сетка? если нет — пересчёт по WCS этого файла
                if data.shape[-2:] != (H, W):
                    x2, y2 = w.world_to_pixel_values(ra_in, dec_in)
                    xi, yi = np.rint(x2).astype(int), np.rint(y2).astype(int)
                    bad = (
                        (xi < CUT)
                        | (xi >= data.shape[-1] - CUT)
                        | (yi < CUT)
                        | (yi >= data.shape[-2] - CUT)
                    )
                    xi[bad], yi[bad] = CUT, CUT  # заглушка для вышедших за край
            for k in range(len(ids_in)):
                patch = np.asarray(
                    data[yi[k] - CUT : yi[k] + CUT, xi[k] - CUT : xi[k] + CUT], dtype=np.float32
                )
                if patch.shape != (2 * CUT, 2 * CUT):
                    patch = np.zeros((2 * CUT, 2 * CUT), dtype=np.float32)
                X[k, BANDS.index(band)] = np.nan_to_num(patch, nan=0.0)
        print(f"  {band}: готово")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        X=X.astype(np.float16),
        ids=ids_in,
        ra=ra_in,
        dec=dec_in,
    )
    print(f"СОХРАНЕНО {OUT}: {X.shape}")


if __name__ == "__main__":
    main()
