from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lib.metrics.bootstrap import Indices, bootstrap_distribution, group_members
from photoz.metrics import photoz_stats

N_BOOT = 2000
TILE_SEED = 20261105
SEED_TILE_SEED = 20261104
ARM_SEED = 20261208
OUTLIER_THRESHOLD = 0.15
MIN_ROWS = 30

type FloatArray = NDArray[np.float64]
type Record = dict[str, object]


def stats_record(pred: FloatArray, z: FloatArray) -> dict[str, float]:
    n_mad, outliers, b = photoz_stats(pred, z)
    return {"nmad": n_mad, "outliers": outliers, "bias": b}


def metric_record(pred: FloatArray, z: FloatArray) -> dict[str, float]:
    return {"n": int(len(z)), **stats_record(pred, z)}


def tile_members(tiles: NDArray[Any]) -> list[NDArray[np.intp]]:
    return group_members(tiles)


def tile_boot(
    a: FloatArray,
    b: FloatArray | None,
    z: FloatArray,
    tiles: NDArray[Any],
    n_boot: int = N_BOOT,
    seed: int = TILE_SEED,
) -> list[float]:
    if not len(z):
        return [np.nan, np.nan]
    mem = tile_members(tiles)

    def difference(draw: Indices) -> float:
        ix = np.concatenate([mem[i] for i in draw])
        base = photoz_stats(b[ix], z[ix])[0] if b is not None else 0.0
        return photoz_stats(a[ix], z[ix])[0] - base

    reps = bootstrap_distribution(difference, (len(mem),), n_boot, seed)
    return [float(v) for v in np.percentile(reps, [2.5, 97.5])]


def seed_tile_boot(
    pa: FloatArray,
    pb: FloatArray,
    z: FloatArray,
    tiles: NDArray[Any],
    n_boot: int = N_BOOT,
    seed: int = SEED_TILE_SEED,
) -> tuple[list[float], list[float]]:
    if not len(z):
        return [np.nan, np.nan], [np.nan, np.nan]
    mem = tile_members(tiles)
    pb = pb[None] if pb.ndim == 1 else pb

    def difference(draw: Indices, seeds_a: Indices, seeds_b: Indices) -> FloatArray:
        ix = np.concatenate([mem[i] for i in draw])
        na, oa, _ = photoz_stats(pa[seeds_a].mean(0)[ix], z[ix])
        nb, ob, _ = photoz_stats(pb[seeds_b].mean(0)[ix], z[ix])
        return np.array([na - nb, oa - ob])

    reps = bootstrap_distribution(difference, (len(mem), len(pa), len(pb)), n_boot, seed)
    bounds = np.percentile(reps, [2.5, 97.5], axis=-1)
    return bounds[:, 0].tolist(), bounds[:, 1].tolist()


def seed_summary(
    pa: FloatArray,
    z: FloatArray,
    tiles: NDArray[Any],
    comps: Mapping[str, FloatArray],
    n_boot: int = N_BOOT,
    seed: int = SEED_TILE_SEED,
) -> Record:
    fin = np.isfinite(pa).all(0) & np.isfinite(z)
    pa, z, tiles = pa[:, fin], z[fin], tiles[fin]
    kept = {k: c[fin] for k, c in comps.items()}
    if len(z) < MIN_ROWS:
        return {"n": int(len(z)), "note": "too few rows"}
    n_mad, outliers, b = photoz_stats(pa.mean(0), z)
    out: Record = {
        "n": int(len(z)),
        "tiles": int(len(np.unique(tiles))),
        "nmad": n_mad,
        "outliers": outliers,
        "bias": b,
        "seed_nmad": [photoz_stats(p, z)[0] for p in pa],
    }
    for name, pc in kept.items():
        ok = np.isfinite(pc)
        if ok.sum() < MIN_ROWS:
            continue
        nm, om, bm = photoz_stats(pa.mean(0)[ok], z[ok])
        nc, oc, bc = photoz_stats(pc[ok], z[ok])
        ci_n, ci_o = seed_tile_boot(pa[:, ok], pc[ok], z[ok], tiles[ok], n_boot, seed)
        out["vs_" + name] = {
            "n": int(ok.sum()),
            "model_nmad": nm,
            "comp_nmad": nc,
            "model_out": om,
            "comp_out": oc,
            "model_bias": bm,
            "comp_bias": bc,
            "d_nmad": nm - nc,
            "d_nmad_ci": ci_n,
            "d_out": om - oc,
            "d_out_ci": ci_o,
        }
    return out


def arm_boot(
    models: Mapping[str, FloatArray],
    comps: Mapping[str, FloatArray],
    z: FloatArray,
    tiles: NDArray[Any],
    n_boot: int = N_BOOT,
    seed: int = ARM_SEED,
) -> dict[str, dict[str, FloatArray]]:
    mem = tile_members(tiles)
    if not models or not comps or n_boot == 0:
        return {m: {c: np.empty((n_boot, 2)) for c in comps} for m in models}
    counts = sorted({len(v) for v in models.values()})

    def differences(draw: Indices, *seeds: Indices) -> FloatArray:
        ix = np.concatenate([mem[i] for i in draw])
        zz = z[ix]
        picks = dict(zip(counts, seeds, strict=True))
        reference = np.array([photoz_stats(v[ix], zz)[:2] for v in comps.values()])
        predicted = np.array([
            photoz_stats(v[picks[len(v)]].mean(0)[ix], zz)[:2] for v in models.values()
        ])
        delta: FloatArray = predicted[:, None, :] - reference
        return delta

    reps = bootstrap_distribution(differences, (len(mem), *counts), n_boot, seed)
    return {
        m: {c: reps[i, j].T for j, c in enumerate(comps)}
        for i, m in enumerate(models)
    }


def arm_summary(
    models: Mapping[str, FloatArray],
    comps: Mapping[str, FloatArray],
    z: FloatArray,
    tiles: NDArray[Any],
    n_boot: int = N_BOOT,
    seed: int = ARM_SEED,
) -> Record:
    if not len(z):
        return {"n": 0, "tiles": 0, "note": "no common finite observations"}
    reps = arm_boot(models, comps, z, tiles, n_boot, seed)
    arms = {m: metric_record(v.mean(0), z) for m, v in models.items()} | {
        c: metric_record(v, z) for c, v in comps.items()
    }
    out: Record = {"n": int(len(z)), "tiles": int(len(np.unique(tiles))), "arms": arms}
    for m in models:
        for c in comps:
            d = reps[m][c]
            out[f"{m}_vs_{c}"] = {
                "d_nmad": arms[m]["nmad"] - arms[c]["nmad"],
                "d_out": arms[m]["outliers"] - arms[c]["outliers"],
                "d_nmad_ci95": np.percentile(d[:, 0], [2.5, 97.5]).tolist(),
                "d_out_ci95": np.percentile(d[:, 1], [2.5, 97.5]).tolist(),
            }
    return out
