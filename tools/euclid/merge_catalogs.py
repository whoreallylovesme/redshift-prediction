#!/usr/bin/env python3
"""Сборка тренировочной таблицы Euclid Q1 EDF-N.

Мердж по OBJECT_ID:
  SPE (жёсткий кат гризм-меток) × MER (морфология/флаги) × PHZ (UNIF-потоки, photo-z бейзлайн).
Тайл берётся из имени SPE-файла; перекрытия тайлов дедуплицируются (первое вхождение).
Выход: datasets/euclid_q1/merged/edfn_labels.parquet
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

# жёсткий кат меток
CUT_PROB = 0.9
CUT_ZERR = 0.01
CUT_SNR = 5.0
Z_MIN, Z_MAX = 0.001, 3.0


def collect_spe() -> pd.DataFrame:
    rows: list[dict] = []
    seen: set[int] = set()
    qso_ids: set[int] = set()
    n_dup = n_raw = 0
    for f in sorted((BASE / "spe").glob("*.fits")):
        tile = int(re.search(r"-(\d{9})_", f.name).group(1))
        with fits.open(f, memmap=True) as hd:
            g = hd["SPE_GALAXY_CANDIDATES"].data
            q = hd["SPE_QSO_CANDIDATES"].data
            qso_ids.update(np.asarray(q["OBJECT_ID"], dtype=np.int64).tolist())
            oid = np.asarray(g["OBJECT_ID"], dtype=np.int64)
            prob = np.asarray(g["SPE_Z_PROB"], dtype=np.float64)
            zerr = np.asarray(g["SPE_Z_ERR"], dtype=np.float64)
            snr = np.asarray(g["SPE_CONT_SNR"], dtype=np.float64)
            z = np.asarray(g["SPE_Z"], dtype=np.float64)
            good = (
                (prob > CUT_PROB) & (zerr < CUT_ZERR) & (snr > CUT_SNR) & (z > Z_MIN) & (z < Z_MAX)
            )
            n_raw += len(g)
            # вероятности классификации (могут покрывать подмножество)
            cls_map: dict[int, tuple[float, float, float]] = {}
            if "SPE_CLASSIFICATION" in hd:
                c = hd["SPE_CLASSIFICATION"].data
                for i, cid in enumerate(np.asarray(c["OBJECT_ID"], dtype=np.int64)):
                    cls_map[cid] = (
                        float(c["SPE_GAL_PROB"][i]),
                        float(c["SPE_QSO_PROB"][i]),
                        float(c["SPE_STAR_PROB"][i]),
                    )
            for i in np.flatnonzero(good):
                o = int(oid[i])
                if o in seen:
                    n_dup += 1
                    continue
                seen.add(o)
                gp, qp, sp = cls_map.get(o, (np.nan, np.nan, np.nan))
                rows.append(
                    {
                        "object_id": o,
                        "tile": tile,
                        "spe_z": z[i],
                        "spe_z_err": zerr[i],
                        "spe_z_prob": prob[i],
                        "spe_cont_snr": snr[i],
                        "spe_gal_prob": gp,
                        "spe_qso_prob": qp,
                        "spe_star_prob": sp,
                        "is_qso_candidate": o in qso_ids,
                    }
                )
    df = pd.DataFrame(rows)
    print(f"SPE: сырых {n_raw:,}, после ката и дедупа {len(df):,} (дубли перекрытий: {n_dup:,})")
    return df


def join_mer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("object_id")
    morph_cols = [
        "RIGHT_ASCENSION",
        "DECLINATION",
        "FWHM",
        "KRON_RADIUS",
        "EXTENDED_PROB",
        "SPURIOUS_PROB",
        "MAG_STARGAL_SEP",
        "DET_QUALITY_FLAG",
    ]
    out = pd.DataFrame(index=df.index)
    out["tile"] = df["tile"]
    matched = pd.Series(False, index=df.index)
    for f in sorted((BASE / "mer").glob("*.fits")):
        with fits.open(f, memmap=True) as hd:
            d = hd[1].data
            oid = np.asarray(d["OBJECT_ID"], dtype=np.int64)
            hit = pd.Index(oid).intersection(df.index)
            if len(hit) == 0:
                continue
            pos = {o: i for i, o in enumerate(oid)}
            idx = [pos[o] for o in hit]
            for col in morph_cols:
                s = pd.Series(np.asarray(d[col], dtype=np.float64)[idx], index=hit)
                prev = out.get(col.lower())
                out[col.lower()] = s.reindex(out.index) if prev is None else prev.fillna(s)
            matched.loc[hit] = True
    df["mer_matched"] = matched.values
    print(f"MER: найдено {int(matched.sum()):,} из {len(df):,}")
    ren = {"RIGHT_ASCENSION": "right_ascension", "DECLINATION": "declination"}
    for col in morph_cols:
        name = ren.get(col, col.lower())
        series = out[col.lower()].reindex(df.index)
        if name in df.columns:
            df[name] = df[name].fillna(series).values
        else:
            df[name] = series.values
    return df.reset_index()


def join_phz(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("object_id")
    flux_bands = [
        "U_EXT_DECAM_UNIF",
        "G_EXT_DECAM_UNIF",
        "R_EXT_DECAM_UNIF",
        "I_EXT_DECAM_UNIF",
        "Z_EXT_DECAM_UNIF",
        "VIS_UNIF",
        "Y_UNIF",
        "J_UNIF",
        "H_UNIF",
    ]
    cols = (
        ["PHZ_MEDIAN", "PHZ_MODE_1", "PHZ_FLAGS", "PHZ_CLASSIFICATION"]
        + [f"FLUX_{b}" for b in flux_bands]
        + [f"FLUXERR_{b}" for b in flux_bands]
    )
    acc: dict[str, pd.Series] = {}
    matched = pd.Series(False, index=df.index)
    for f in sorted((BASE / "phz").glob("*.fits")):
        with fits.open(f, memmap=True) as hd:
            d = hd[1].data
            oid = np.asarray(d["OBJECT_ID"], dtype=np.int64)
            hit = pd.Index(oid).intersection(df.index)
            if len(hit) == 0:
                continue
            pos = {o: i for i, o in enumerate(oid)}
            idx = np.array([pos[o] for o in hit])
            for col in cols:
                s = pd.Series(np.asarray(d[col], dtype=np.float64)[idx], index=hit)
                if col not in acc:
                    acc[col] = s.reindex(df.index)
                else:
                    acc[col] = acc[col].fillna(s)
            matched.loc[hit] = True
    df["phz_matched"] = matched.values
    print(f"PHZ: найдено {int(matched.sum()):,} из {len(df):,}")
    ren = {
        "PHZ_MEDIAN": "phz_median",
        "PHZ_MODE_1": "phz_mode1",
        "PHZ_FLAGS": "phz_flags",
        "PHZ_CLASSIFICATION": "phz_class",
    }
    for col, series in acc.items():
        name = ren.get(col, col.lower())
        df[name] = series.values
    return df.reset_index()


def main() -> None:
    df = collect_spe()
    df = join_mer(df)
    df = join_phz(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\nсохранено: {OUT} ({len(df):,} строк, {df.shape[1]} колонок)")
    print("\n=== сводка ===")
    print(f"QSO-кандидаты среди меток: {int(df['is_qso_candidate'].sum()):,}")
    print(f"spe_qso_prob > 0.5: {(df['spe_qso_prob'] > 0.5).sum():,}")
    print(f"тайлов: {df['tile'].nunique()}")
    print(
        f"z: медиана {df['spe_z'].median():.3f}, "
        f"перцентили 5/95: {df['spe_z'].quantile(0.05):.2f}/{df['spe_z'].quantile(0.95):.2f}"
    )
    print(f"PHZ_MEDIAN доступен: {df['phz_median'].notna().sum():,}")
    nm = df["phz_median"].notna()
    if nm.any():
        dz = (df.loc[nm, "phz_median"] - df.loc[nm, "spe_z"]) / (1 + df.loc[nm, "spe_z"])
        nmad = 1.4826 * np.median(np.abs(dz - np.median(dz)))
        out = (np.abs(dz) > 0.15).mean() * 100
        print(
            f"бейзлайн PHZ vs гризм: NMAD={nmad:.4f}, bias={dz.median():+.4f}, выбросы={out:.2f}%"
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="каталог с spe/ mer/ phz/")
    ap.add_argument("--out", default=None, help="путь итогового parquet")
    args = ap.parse_args()
    BASE = Path(args.base)
    OUT = Path(args.out) if args.out else BASE / "merged" / "edfn_labels.parquet"
    main()
