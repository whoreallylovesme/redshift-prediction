from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from lib.dataloaders.euclid_translation import (
    PUBLISHED_ZPC_S,
    SOUTH_FIT_SLOTS,
    Grids,
    impute_ebv,
    load_labels,
    load_photometry,
    profile_fit,
    south_inputs,
)
from lib.utils.log import get_logger
from lib.utils.utils import check

logger = get_logger()

type F64 = NDArray[np.float64]
type I64 = NDArray[np.int64]
type Bool = NDArray[np.bool_]
type Draws = tuple[NDArray[Any], I64, I64]

NAMES = ("g", "r", "i", "z", "VIS", "Y", "J", "H")
EUCLID = [4, 5, 6, 7]
ALL_BANDS = tuple(range(8))
SN_MIN = 20.0
MIN_FIT = 4
NBOOT = 1000
SEED = 20261105
RULE = "all7"
OUTPUT = "zpc_f.json"


class ZeroPointConfig(BaseModel):
    labels: Path
    f_train_photometry: Path
    f_valid_photometry: Path
    grids_dir: Path
    out_dir: Path
    nested: int = 0


class BandFit(BaseModel):
    n: int
    n_tiles: int | None = None
    zpc_in: float | None = None
    factor: float | None = None
    factor_ci95: list[float] | None = None
    factor_boot_sd: float | None = None
    median_dmag: float | None = None
    median_dmag_ci95: list[float] | None = None
    nmad_dmag: float | None = None


@dataclass(frozen=True)
class Sample:
    object_id: I64
    flux: F64
    err: F64
    valid: Bool
    z: F64
    ebv: F64
    tile: I64

    def subset(self, mask: Bool) -> Sample:
        return Sample(
            self.object_id[mask],
            self.flux[mask],
            self.err[mask],
            self.valid[mask],
            self.z[mask],
            self.ebv[mask],
            self.tile[mask],
        )


def load_set(
    name: str, labels: Mapping[str, NDArray[Any]], path: Path
) -> tuple[Sample, dict[str, int]]:
    d = load_photometry(path)
    check(
        np.array_equal(d["object_id"], labels[f"{name}__object_id"])
        and bool((d["field_code"] == 1).all()),
        f"{name} alignment",
    )
    ebv, imputed = impute_ebv(d["ancillary_values"][:, 0], d["tile"])
    z = labels[f"{name}__z_fix"].astype(np.float64)
    keep = np.isfinite(z) & (z >= 0) & (z <= 6)
    sample = Sample(
        object_id=d["object_id"][keep],
        flux=d["flux"][keep][:, SOUTH_FIT_SLOTS],
        err=d["error"][keep][:, SOUTH_FIT_SLOTS],
        valid=d["valid"][keep][:, SOUTH_FIT_SLOTS],
        z=z[keep],
        ebv=ebv[keep],
        tile=d["tile"][keep],
    )
    return sample, dict(rows=len(z), kept=int(keep.sum()), ebv_imputed=int(imputed.sum()))


def predict(
    d: Sample, zpc: F64, grids: Grids, bands: Sequence[int] = ALL_BANDS, chunk: int = 1024
) -> tuple[F64, Bool]:
    f, s, v = south_inputs(d.flux, d.err, d.valid, zpc)
    with np.errstate(divide="ignore", invalid="ignore"):
        hi = v & (d.flux / d.err >= SN_MIN)
    n = len(d.z)
    pred = np.full((n, 8), np.nan)
    for st in range(0, n, chunk):
        sl = slice(st, min(n, st + chunk))
        ms = grids.at_z(d.z[sl], d.ebv[sl])[0]
        r = np.arange(sl.stop - sl.start)
        for b in bands:
            vb = v[sl].copy()
            vb[:, b] = False
            alpha, chi2, ok = profile_fit(
                np.where(vb, f[sl], 0.0), np.where(vb, s[sl], 1.0), vb, ms
            )
            kb = np.argmin(chi2, axis=1)
            p = alpha[r, kb] * ms[r, kb, b]
            pred[sl, b] = np.where((vb.sum(1) >= MIN_FIT) & ok.any(1), p, np.nan)
    return pred, hi


def gate(pred: F64, hi: Bool) -> Bool:
    use = np.zeros(pred.shape, bool)
    for b in range(8):
        others = np.delete(np.arange(8), b)
        fit_ok = hi[:, others].all(1)
        use[:, b] = hi[:, b] & fit_ok & np.isfinite(pred[:, b]) & (pred[:, b] > 0)
    return use


def run(d: Sample, zpc: F64, grids: Grids, bands: Sequence[int] = ALL_BANDS) -> tuple[F64, Bool]:
    pred, hi = predict(d, zpc, grids, bands=bands)
    return pred, gate(pred, hi)


def tile_draws(tile: I64, nboot: int, seed: int) -> Draws:
    tu, ti = np.unique(tile, return_inverse=True)
    rng = np.random.default_rng(seed)
    cnt = np.stack(
        [np.bincount(rng.integers(0, len(tu), len(tu)), minlength=len(tu)) for _ in range(nboot)]
    )
    return tu, ti, cnt


def boot_median(x: F64, ti: I64, cnt: I64) -> F64:
    return np.array([np.median(np.repeat(x, w[ti])) if w[ti].sum() else np.nan for w in cnt])


def nmad(x: F64) -> float | None:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else None


def summarize(d: Sample, pred: F64, use: Bool, zpc: F64, draws: Draws) -> dict[str, BandFit]:
    _, ti, cnt = draws
    res: dict[str, BandFit] = {}
    for b in range(8):
        u = use[:, b]
        if not u.any():
            res[NAMES[b]] = BandFit(n=0)
            continue
        ratio = pred[u, b] / d.flux[u, b]
        dm = -2.5 * np.log10(d.flux[u, b] * zpc[b] / pred[u, b])
        bf, bd = boot_median(ratio, ti[u], cnt), boot_median(dm, ti[u], cnt)
        res[NAMES[b]] = BandFit(
            n=int(u.sum()),
            n_tiles=int(len(np.unique(ti[u]))),
            zpc_in=float(zpc[b]),
            factor=float(np.median(ratio)),
            factor_ci95=np.nanquantile(bf, [0.025, 0.975]).tolist(),
            factor_boot_sd=float(np.nanstd(bf)),
            median_dmag=float(np.median(dm)),
            median_dmag_ci95=np.nanquantile(bd, [0.025, 0.975]).tolist(),
            nmad_dmag=nmad(dm),
        )
    return res


def factor_of(fit: BandFit) -> float:
    if fit.factor is None:
        raise ValueError("zero-point factor needs at least one usable row")
    return fit.factor


def euclid_of(s: Mapping[str, BandFit]) -> F64:
    return np.array([factor_of(s[NAMES[b]]) for b in EUCLID])


def with_euclid(f: F64) -> F64:
    z = PUBLISHED_ZPC_S.copy()
    z[EUCLID] = f
    return z


def nested(
    d: Sample,
    rows: Mapping[int, F64],
    grids: Grids,
    use0: Bool,
    ti: I64,
    cnt: I64,
) -> F64:
    out: list[list[float]] = []
    for k, w in enumerate(cnt):
        f1 = np.array([np.median(np.repeat(rows[b], w[ti[use0[:, b]]])) for b in EUCLID])
        pred, use = run(d, with_euclid(f1), grids, bands=EUCLID)
        out.append(
            [
                float(
                    np.median(
                        np.repeat(pred[use[:, b], b] / d.flux[use[:, b], b], w[ti[use[:, b]]])
                    )
                )
                for b in EUCLID
            ]
        )
        if k % 50 == 0:
            logger.info(f"nested {k} {np.round(out[-1], 4).tolist()}")
    return np.array(out)


def _plain(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(exclude_none=True)
    if isinstance(value, np.ndarray | np.generic):
        return value.tolist()
    raise TypeError(type(value).__name__)


def fit_zero_points(config: ZeroPointConfig) -> Path:
    grids = Grids.load(config.grids_dir)
    labels = load_labels(config.labels)
    ft, ft_info = load_set("F_TRAIN", labels, config.f_train_photometry)
    fv, fv_info = load_set("F_VALID", labels, config.f_valid_photometry)
    on_valid = np.isin(ft.tile, np.unique(fv.tile))
    excluded = int(on_valid.sum())
    ft = ft.subset(~on_valid)
    draws = tile_draws(ft.tile, NBOOT, SEED)
    logger.info(f"F_TRAIN {ft_info['kept']} on F_VALID tiles {excluded} tiles {len(draws[0])}")
    p1, u1 = run(ft, PUBLISHED_ZPC_S, grids)
    s1 = summarize(ft, p1, u1, PUBLISHED_ZPC_S, draws)
    f1 = euclid_of(s1)
    z2 = with_euclid(f1)
    p2, u2 = run(ft, z2, grids)
    s2 = summarize(ft, p2, u2, z2, draws)
    f2 = euclid_of(s2)
    zf = with_euclid(f2)
    logger.info(f"{RULE} iter1 {np.round(f1, 4).tolist()} iter2 {np.round(f2, 4).tolist()}")
    nest = None
    if config.nested:
        m = np.asarray(u1.any(1), dtype=bool)
        rows = {b: p1[u1[:, b], b] / ft.flux[u1[:, b], b] for b in EUCLID}
        nb = nested(ft.subset(m), rows, grids, u1[m], draws[1][m], draws[2][: config.nested])
        nest = dict(
            nboot=int(config.nested),
            rule=RULE,
            ci95={
                NAMES[b]: np.nanquantile(nb[:, i], [0.025, 0.975]).tolist()
                for i, b in enumerate(EUCLID)
            },
            sd={NAMES[b]: float(np.nanstd(nb[:, i])) for i, b in enumerate(EUCLID)},
        )
    euclid = {NAMES[b]: float(zf[b]) for b in EUCLID}
    result = dict(
        primary_rule=RULE,
        published_zpc_s=PUBLISHED_ZPC_S,
        sn_min=SN_MIN,
        min_fit_bands=MIN_FIT,
        nboot=NBOOT,
        seed=SEED,
        factors=dict(
            euclid=euclid,
            euclid_ci95={NAMES[b]: s2[NAMES[b]].factor_ci95 for b in EUCLID},
            euclid_iter1={NAMES[b]: float(f1[i]) for i, b in enumerate(EUCLID)},
            euclid_iter1_ci95={NAMES[b]: s1[NAMES[b]].factor_ci95 for b in EUCLID},
            decam_sanity={NAMES[b]: s2[NAMES[b]].factor for b in range(4)},
            decam_sanity_iter1={NAMES[b]: s1[NAMES[b]].factor for b in range(4)},
            published_decam=PUBLISHED_ZPC_S[:4],
            published_euclid=PUBLISHED_ZPC_S[4:],
            n={NAMES[b]: s2[NAMES[b]].n for b in range(8)},
        ),
        iterations=[
            dict(iteration=1, zpc_in=PUBLISHED_ZPC_S, bands=s1),
            dict(iteration=2, zpc_in=z2, bands=s2),
        ],
        nested_bootstrap=nest,
        sample=dict(
            F_TRAIN_rows=ft_info["rows"],
            F_TRAIN_used=int(len(ft.z)),
            F_TRAIN_rows_on_F_VALID_tiles_excluded=excluded,
            F_TRAIN_tiles=int(len(draws[0])),
            F_VALID_rows=fv_info["kept"],
            F_VALID_tiles=int(len(np.unique(fv.tile))),
            ebv_imputed=dict(F_TRAIN=ft_info["ebv_imputed"], F_VALID=fv_info["ebv_imputed"]),
        ),
    )
    config.out_dir.mkdir(parents=True, exist_ok=True)
    path = config.out_dir / OUTPUT
    path.write_text(json.dumps(result, indent=1, allow_nan=True, default=_plain) + "\n")
    logger.info(f"euclid factors {euclid} -> {path}")
    return path
