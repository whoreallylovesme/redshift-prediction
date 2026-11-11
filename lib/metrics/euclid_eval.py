from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from lib.dataloaders.euclid_report import (
    PRIMARY_STRATUM,
    EvalBlock,
    SeedStack,
    SetPair,
    block_values,
    fill_width,
)
from lib.metrics.bootstrap import (
    Record,
    arm_summary,
    photoz_stats,
    seed_summary,
    stats_record,
    tile_boot,
)

HYBRID_WEIGHT = 0.5
COMPARATORS = ("ml", "ppub", "zpf", "poff")
CLASSICAL_N_SETS: tuple[tuple[str, tuple[str, float] | None], ...] = (
    ("N_all", None),
    ("N_vis_22.5_30", ("vis", 22.5)),
    ("N_z_1.0_9", ("z", 1.0)),
)

type FloatArray = NDArray[np.float64]
type Variant = dict[str, SetPair]


def fixed_hybrid(z: FloatArray, pp: FloatArray, w: float = HYBRID_WEIGHT) -> FloatArray:
    ok = np.isfinite(pp)
    mix = np.expm1(
        w * np.log1p(np.clip(z, 0, None)) + (1 - w) * np.log1p(np.where(ok, pp, 0))[None]
    )
    out: FloatArray = np.where(ok[None], mix, np.nan)
    return out


def ivw_hybrid(z: FloatArray, s: FloatArray, pp: FloatArray, sp: FloatArray) -> FloatArray:
    w = sp[None] ** 2 / (2 * np.exp(2 * s) + sp[None] ** 2)
    ok = np.isfinite(pp)
    zi = np.expm1(w * np.log1p(np.clip(z, 0, None)) + (1 - w) * np.log1p(np.where(ok, pp, 0))[None])
    out: FloatArray = np.where(ok[None], zi, np.nan)
    return out


def source_metrics(d: pd.DataFrame, methods: Sequence[str], sources: Sequence[str]) -> Record:
    subsets = [("ALL", np.ones(len(d), bool)), ("ALL_z1", d.z.values >= 1.0)]
    subsets += [(s, (d.src == s).values) for s in sources]
    rec: Record = {}
    for name, m in subsets:
        z, t = d.z.values[m], d.tile.values[m]
        r: dict[str, Any] = {"n": int(m.sum())}
        for k in methods:
            r[k] = stats_record(d[k].values[m], z)
            if name == "ALL":
                r[k]["nmad_ci"] = tile_boot(d[k].values[m], None, z, t)
        for k in methods[1:]:
            r[f"d_vs_{k}"] = {
                "d_nmad": r["ml"]["nmad"] - r[k]["nmad"],
                "ci": tile_boot(d.ml.values[m], d[k].values[m], z, t),
            }
        rec[name] = r
    return rec


def method_block(
    frame: Mapping[str, FloatArray], z: FloatArray, tiles: NDArray[Any], cols: Sequence[str]
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for k in cols:
        v = frame[k]
        rec: dict[str, Any] = {**stats_record(v, z), "nmad_ci": tile_boot(v, None, z, tiles)}
        if k != "ml":
            rec["d_ml_minus_this"] = photoz_stats(frame["ml"], z)[0] - rec["nmad"]
            rec["d_ml_minus_this_ci"] = tile_boot(frame["ml"], v, z, tiles)
        out[k] = rec
    return out


def classical_metrics(
    d: pd.DataFrame,
    keys: Sequence[str],
    sets: Sequence[tuple[str, Mapping[str, FloatArray], FloatArray, NDArray[Any]]],
) -> Record:
    cols = [*COMPARATORS, *keys]
    out: dict[str, Any] = {}
    for name, cut in CLASSICAL_N_SETS:
        m = np.ones(len(d), bool) if cut is None else d[cut[0]].values >= cut[1]
        north = {k: d[k].values[m] for k in cols}
        out[name] = {
            "n": int(m.sum()),
            "methods": method_block(north, d.z.values[m], d.tile.values[m], cols),
        }
    for name, frame, z, tiles in sets:
        extra = [k for k in COMPARATORS[1:] if k in frame]
        out[name] = {
            "n": int(len(z)),
            "methods": method_block(frame, z, tiles, ["ml", *extra, *keys]),
        }
    return {"n": int(len(d)), "keys": list(keys), "sets": out}


def zval_metrics(z: FloatArray, methods: Mapping[str, FloatArray]) -> dict[str, Any]:
    rec: dict[str, Any] = {"n": int(len(z))}
    for k, v in methods.items():
        ok = np.isfinite(v)
        rec[k] = {"n": int(ok.sum()), **stats_record(v[ok], z[ok])}
    return rec


def variants(
    preds: Mapping[str, SeedStack],
    points: Mapping[int, float],
    widths: Mapping[int, float],
) -> dict[str, Variant]:
    plain: Variant = {}
    fixed: Variant = {}
    ivw: Variant = {}
    for name, stack in preds.items():
        ppub = np.array([points.get(int(o), np.nan) for o in stack.ids])
        plain[name] = (stack.z, ppub)
        fixed[name] = (fixed_hybrid(stack.z, ppub), ppub)
        sp = fill_width(np.array([widths.get(int(o), np.nan) for o in stack.ids]))
        ivw[name] = (ivw_hybrid(stack.z, stack.s, ppub, sp), ppub)
    return {"": plain, "+ppub": fixed, "+ppub_ivw": ivw}


def block_metrics(blocks: Sequence[EvalBlock], variant: Mapping[str, SetPair]) -> Record:
    rep: Record = {}
    for block in blocks:
        z_model, ppub = block_values(block, variant)
        rep[block.name] = seed_summary(
            z_model, block.z, block.tiles, {"phz": block.phz, "ppub": ppub}
        )
    return rep


def comparison_results(
    tag: str,
    preds: Mapping[str, SeedStack],
    blocks: Sequence[EvalBlock],
    points: Mapping[int, float],
    widths: Mapping[int, float],
) -> Record:
    return {
        f"{tag}{suffix}": block_metrics(blocks, variant)
        for suffix, variant in variants(preds, points, widths).items()
    }


def hetdex_results(
    tag: str,
    h6: SeedStack,
    z: FloatArray,
    tiles: NDArray[Any],
    stratum: NDArray[Any],
    comps: Mapping[str, FloatArray],
    sp: FloatArray,
) -> Record:
    ppub = comps["P_pub"]
    models = {f"{tag}_pure": h6.z, f"{tag}_hybrid": ivw_hybrid(h6.z, h6.s, ppub, sp)}
    fin = np.isfinite(z) & (z > 0)
    for c in comps.values():
        fin &= np.isfinite(c)
    for v in models.values():
        fin &= np.isfinite(v).all(0)
    m = fin & (stratum == PRIMARY_STRATUM)
    return arm_summary(
        {k: v[:, m] for k, v in models.items()},
        {k: v[m] for k, v in comps.items()},
        z[m],
        tiles[m],
    )
