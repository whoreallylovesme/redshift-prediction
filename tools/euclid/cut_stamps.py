#!/usr/bin/env python3
"""Нарезка 64x64x4 штампов из мозаик тайла (VIS,Y,J,H на одной сетке).

Аргументы CLI: tile_id objects.parquet (колонки object_id, right_ascension, declination),
--mosaics (по умолчанию /root/euclid_q1/mosaics/{tile}), --out (по умолчанию
/root/euclid_q1/stamps/{tile}.npz; X float16 [N,4,64,64], ids int64,
ra, dec; каналы: VIS, Y, J, H).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

from lib.utils.log import get_logger

logger = get_logger()

CUT = 32  # полукадр 64x64

BANDS = {"vis": "vis", "y": "nir_y", "j": "nir_j", "h": "nir_h"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """tile и parquet объектов — позиционные; мозаики и выход — с дефолтами."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tile", help="идентификатор тайла")
    ap.add_argument(
        "objects", type=Path, help="parquet с колонками object_id, right_ascension, declination"
    )
    ap.add_argument("--mosaics", type=Path, default=None,
                    help="каталог мозаик (по умолчанию /root/euclid_q1/mosaics/{tile})")
    ap.add_argument("--out", type=Path, default=None,
                    help="итоговый npz (по умолчанию /root/euclid_q1/stamps/{tile}.npz)")
    return ap.parse_args(argv)


def _read_objects(obj: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ra, dec и object_id из каталога объектов."""
    df = pd.read_parquet(obj)
    ra = df["right_ascension"].to_numpy(float)
    dec = df["declination"].to_numpy(float)
    ids = df["object_id"].to_numpy(np.int64)
    return ra, dec, ids


def _world_to_pixels(w: WCS, ra: np.ndarray, dec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Округлённые пиксельные координаты объектов по WCS мозаики."""
    x, y = w.world_to_pixel_values(ra, dec)
    return np.rint(x).astype(int), np.rint(y).astype(int)


def _refit_to_grid(
    w: WCS, ra_in: np.ndarray, dec_in: np.ndarray, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Координаты отобранных объектов на чужой сетке; вышедшим за край — заглушка CUT."""
    x2, y2 = w.world_to_pixel_values(ra_in, dec_in)
    xi = np.rint(x2).astype(int)
    yi = np.rint(y2).astype(int)
    bad = (xi < CUT) | (xi >= shape[-1] - CUT) | (yi < CUT) | (yi >= shape[-2] - CUT)
    xi[bad], yi[bad] = CUT, CUT  # заглушка для вышедших за край
    return xi, yi


def _fill_channel(
    stamps: np.ndarray, data: np.ndarray, xi: np.ndarray, yi: np.ndarray,
    ids: np.ndarray, channel: int,
) -> None:
    """Записать патч 2*CUT x 2*CUT каждого объекта в канал; неполный патч с края — нули."""
    for k in range(len(ids)):
        patch = np.asarray(
            data[yi[k] - CUT : yi[k] + CUT, xi[k] - CUT : xi[k] + CUT], dtype=np.float32
        )
        if patch.shape != (2 * CUT, 2 * CUT):
            patch = np.zeros((2 * CUT, 2 * CUT), dtype=np.float32)
        stamps[k, channel] = np.nan_to_num(patch, nan=0.0)


def _save_stamps(
    out: Path, stamps: np.ndarray, ids: np.ndarray, ra: np.ndarray, dec: np.ndarray
) -> None:
    """Сохранить X (float16) + ids/ra/dec в сжатый npz."""
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=stamps.astype(np.float16), ids=ids, ra=ra, dec=dec)
    logger.info(f"СОХРАНЕНО {out}: {stamps.shape}")


def main() -> None:
    args = parse_args()
    mosaics = (
        args.mosaics if args.mosaics is not None else Path(f"/root/euclid_q1/mosaics/{args.tile}")
    )
    out = args.out if args.out is not None else Path(f"/root/euclid_q1/stamps/{args.tile}.npz")

    ra, dec, ids = _read_objects(args.objects)
    n = len(ids)

    shape: tuple[int, ...] | None = None
    for ch, (band, prefix) in enumerate(BANDS.items()):
        f = mosaics / f"{prefix}.fits"
        with fits.open(f, memmap=True) as hd:
            hdr = hd[0].header if hd[0].data is not None else hd[1].header
            data = hd[0].data if hd[0].data is not None else hd[1].data
            w = WCS(hdr).celestial
            xi, yi = _world_to_pixels(w, ra, dec)
            if shape is None:
                shape = data.shape
                H, W = shape[-2], shape[-1]
                inside = (xi >= CUT) & (xi < W - CUT) & (yi >= CUT) & (yi < H - CUT)
                logger.info(f"{args.tile}: объектов {n}, внутри кадра {inside.sum()}")
                xi, yi = xi[inside], yi[inside]
                ids_in = ids[inside]
                ra_in, dec_in = ra[inside], dec[inside]
                stamps = np.zeros((inside.sum(), 4, 2 * CUT, 2 * CUT), dtype=np.float32)
            else:
                # та же сетка? если нет — пересчёт по WCS этого файла
                if data.shape[-2:] != (H, W):
                    xi, yi = _refit_to_grid(w, ra_in, dec_in, data.shape)
            _fill_channel(stamps, data, xi, yi, ids_in, ch)
        logger.info(f"  {band}: готово")

    _save_stamps(out, stamps, ids_in, ra_in, dec_in)


if __name__ == "__main__":
    main()
