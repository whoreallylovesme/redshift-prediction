#!/usr/bin/env python3
"""Сборка тренировочной таблицы Euclid Q1 EDF-N.

Мердж по OBJECT_ID:
  SPE (жёсткий кат гризм-меток) × MER (морфология/флаги) × PHZ (UNIF-потоки, photo-z бейзлайн).
Тайл берётся из имени SPE-файла; перекрытия тайлов дедуплицируются (первое вхождение).
Пороги ката меток и выходной путь задаются CLI (дефолты — константы модуля).
Выход: datasets/euclid_q1/merged/edfn_labels.parquet
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.typing import NDArray

from lib.metrics import bias, nmad, outlier_percent
from lib.utils.log import get_logger

logger = get_logger()

# жёсткий кат меток (дефолты CLI-порогов)
CUT_PROB = 0.9
CUT_ZERR = 0.01
CUT_SNR = 5.0
Z_MIN, Z_MAX = 0.001, 3.0


@dataclass(frozen=True)
class LabelCuts:
    """Пороги жёсткого ката гризм-меток (переопределяются через CLI)."""

    prob: float = CUT_PROB
    zerr: float = CUT_ZERR
    snr: float = CUT_SNR
    z_min: float = Z_MIN
    z_max: float = Z_MAX


def parse_args() -> argparse.Namespace:
    """Пути и пороги ката меток; дефолты порогов — константы модуля."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, help="каталог с spe/ mer/ phz/")
    ap.add_argument("--out", default=None, help="путь итогового parquet")
    ap.add_argument("--cut-prob", type=float, default=CUT_PROB, help="порог SPE_Z_PROB")
    ap.add_argument("--cut-zerr", type=float, default=CUT_ZERR, help="порог SPE_Z_ERR")
    ap.add_argument("--cut-snr", type=float, default=CUT_SNR, help="порог SPE_CONT_SNR")
    ap.add_argument("--z-min", type=float, default=Z_MIN, help="нижняя граница SPE_Z")
    ap.add_argument("--z-max", type=float, default=Z_MAX, help="верхняя граница SPE_Z")
    return ap.parse_args()


def _tile_from_name(name: str) -> int:
    """Тайл из имени SPE-файла (фрагмент -NNNNNNNNN_)."""
    m = re.search(r"-(\d{9})_", name)
    if m is None:
        raise ValueError(f"не удалось извлечь тайл из имени файла: {name}")
    return int(m.group(1))


def _label_mask(
    prob: NDArray[np.float64],
    zerr: NDArray[np.float64],
    snr: NDArray[np.float64],
    z: NDArray[np.float64],
    cuts: LabelCuts,
) -> NDArray[np.bool_]:
    """Жёсткий кат: вероятность, ошибка z, SNR континуума и диапазон z."""
    return (
        (prob > cuts.prob)
        & (zerr < cuts.zerr)
        & (snr > cuts.snr)
        & (z > cuts.z_min)
        & (z < cuts.z_max)
    )


def _classification_probs(hd: fits.HDUList) -> dict[int, tuple[float, float, float]]:
    """SPE_CLASSIFICATION → {OBJECT_ID: (gal, qso, star)}; HDU может отсутствовать."""
    cls_map: dict[int, tuple[float, float, float]] = {}
    if "SPE_CLASSIFICATION" in hd:
        c = hd["SPE_CLASSIFICATION"].data
        gal = np.asarray(c["SPE_GAL_PROB"], dtype=np.float64)
        qso = np.asarray(c["SPE_QSO_PROB"], dtype=np.float64)
        star = np.asarray(c["SPE_STAR_PROB"], dtype=np.float64)
        for row, cid in enumerate(np.asarray(c["OBJECT_ID"], dtype=np.int64)):
            cls_map[cid] = (float(gal[row]), float(qso[row]), float(star[row]))
    return cls_map


def _label_rows(
    tile: int,
    oid: NDArray[np.int64],
    z: NDArray[np.float64],
    zerr: NDArray[np.float64],
    prob: NDArray[np.float64],
    snr: NDArray[np.float64],
    good: NDArray[np.bool_],
    cls_map: dict[int, tuple[float, float, float]],
    qso_ids: set[int],
    seen: set[int],
) -> tuple[list[dict[str, object]], int]:
    """Строки меток по маске good с дедупом перекрытий тайлов → (rows, n_dup)."""
    rows: list[dict[str, object]] = []
    n_dup = 0
    for gi in np.flatnonzero(good):
        o = int(oid[gi])
        if o in seen:
            n_dup += 1
            continue
        seen.add(o)
        gp, qp, sp = cls_map.get(o, (np.nan, np.nan, np.nan))
        rows.append(
            {
                "object_id": o,
                "tile": tile,
                "spe_z": z[gi],
                "spe_z_err": zerr[gi],
                "spe_z_prob": prob[gi],
                "spe_cont_snr": snr[gi],
                "spe_gal_prob": gp,
                "spe_qso_prob": qp,
                "spe_star_prob": sp,
                "is_qso_candidate": o in qso_ids,
            }
        )
    return rows, n_dup


def collect_spe(base: Path, cuts: LabelCuts) -> pd.DataFrame:
    """Жёсткий кат гризм-меток по всем SPE-тайлам (дедуп перекрытий — первое вхождение)."""
    rows: list[dict[str, object]] = []
    seen: set[int] = set()
    qso_ids: set[int] = set()
    n_dup = n_raw = 0
    for f in sorted((base / "spe").glob("*.fits")):
        tile = _tile_from_name(f.name)
        with fits.open(f, memmap=True) as hd:
            g = hd["SPE_GALAXY_CANDIDATES"].data
            q = hd["SPE_QSO_CANDIDATES"].data
            qso_ids.update(np.asarray(q["OBJECT_ID"], dtype=np.int64).tolist())
            oid = np.asarray(g["OBJECT_ID"], dtype=np.int64)
            prob = np.asarray(g["SPE_Z_PROB"], dtype=np.float64)
            zerr = np.asarray(g["SPE_Z_ERR"], dtype=np.float64)
            snr = np.asarray(g["SPE_CONT_SNR"], dtype=np.float64)
            z = np.asarray(g["SPE_Z"], dtype=np.float64)
            good = _label_mask(prob, zerr, snr, z, cuts)
            n_raw += len(g)
            # вероятности классификации (могут покрывать подмножество)
            cls_map = _classification_probs(hd)
            file_rows, file_dup = _label_rows(
                tile, oid, z, zerr, prob, snr, good, cls_map, qso_ids, seen
            )
            rows.extend(file_rows)
            n_dup += file_dup
    df = pd.DataFrame(rows)
    logger.info(
        f"SPE: сырых {n_raw:,}, после ката и дедупа {len(df):,} (дубли перекрытий: {n_dup:,})"
    )
    return df


def _join_by_object_id(
    base: Path,
    subdir: str,
    index: pd.Index,
    cols: list[str],
) -> tuple[dict[str, pd.Series], pd.Series]:
    """Мердж колонок из всех FITS-файлов subdir по OBJECT_ID.

    Перекрытия файлов разрешаются fillna: первое ненулевое вхождение приоритетно.
    Возвращает серии по колонкам на полном index и маску matched.
    """
    acc: dict[str, pd.Series] = {}
    matched = pd.Series(False, index=index)
    for f in sorted((base / subdir).glob("*.fits")):
        with fits.open(f, memmap=True) as hd:
            d = hd[1].data
            oid = np.asarray(d["OBJECT_ID"], dtype=np.int64)
            hit = pd.Index(oid).intersection(index)
            if len(hit) == 0:
                continue
            pos = {o: i for i, o in enumerate(oid)}
            idx = [pos[o] for o in hit]
            for col in cols:
                s = pd.Series(np.asarray(d[col], dtype=np.float64)[idx], index=hit)
                acc[col] = s.reindex(index) if col not in acc else acc[col].fillna(s)
            matched.loc[hit] = True
    return acc, matched


def join_mer(df: pd.DataFrame, base: Path) -> pd.DataFrame:
    """Морфология и флаги качества из MER-каталога."""
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
    acc, matched = _join_by_object_id(base, "mer", df.index, morph_cols)
    df["mer_matched"] = matched.values
    logger.info(f"MER: найдено {int(matched.sum()):,} из {len(df):,}")
    ren = {"RIGHT_ASCENSION": "right_ascension", "DECLINATION": "declination"}
    by_lower = {c.lower(): s for c, s in acc.items()}
    for col in morph_cols:
        name = ren.get(col, col.lower())
        series = by_lower[col.lower()].reindex(df.index)
        if name in df.columns:
            df[name] = df[name].fillna(series).values
        else:
            df[name] = series.values
    return df.reset_index()


def join_phz(df: pd.DataFrame, base: Path) -> pd.DataFrame:
    """UNIF-потоки и photo-z бейзлайн из PHZ-каталога."""
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
    acc, matched = _join_by_object_id(base, "phz", df.index, cols)
    df["phz_matched"] = matched.values
    logger.info(f"PHZ: найдено {int(matched.sum()):,} из {len(df):,}")
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


def main(base: Path, out: Path, cuts: LabelCuts) -> None:
    """Полный пайплайн: SPE-метки → мердж MER и PHZ → parquet + сводка."""
    df = collect_spe(base, cuts)
    df = join_mer(df, base)
    df = join_phz(df, base)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info(f"\nсохранено: {out} ({len(df):,} строк, {df.shape[1]} колонок)")
    logger.info("\n=== сводка ===")
    logger.info(f"QSO-кандидаты среди меток: {int(df['is_qso_candidate'].sum()):,}")
    logger.info(f"spe_qso_prob > 0.5: {(df['spe_qso_prob'] > 0.5).sum():,}")
    logger.info(f"тайлов: {df['tile'].nunique()}")
    logger.info(
        f"z: медиана {df['spe_z'].median():.3f}, "
        f"перцентили 5/95: {df['spe_z'].quantile(0.05):.2f}/{df['spe_z'].quantile(0.95):.2f}"
    )
    logger.info(f"PHZ_MEDIAN доступен: {df['phz_median'].notna().sum():,}")
    nm = df["phz_median"].notna()
    if nm.any():
        y_true = df.loc[nm, "spe_z"]
        y_pred = df.loc[nm, "phz_median"].to_numpy(dtype=np.float64)
        logger.info(
            f"бейзлайн PHZ vs гризм: NMAD={nmad(y_true, y_pred):.4f}, "
            f"bias={bias(y_true, y_pred):+.4f}, выбросы={outlier_percent(y_true, y_pred):.2f}%"
        )


if __name__ == "__main__":
    args = parse_args()
    base_dir = Path(args.base)
    out_file = Path(args.out) if args.out else base_dir / "merged" / "edfn_labels.parquet"
    main(
        base_dir,
        out_file,
        LabelCuts(
            prob=args.cut_prob,
            zerr=args.cut_zerr,
            snr=args.cut_snr,
            z_min=args.z_min,
            z_max=args.z_max,
        ),
    )
