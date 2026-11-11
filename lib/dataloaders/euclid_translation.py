from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.typing import NDArray
from pydantic import BaseModel

from lib.dataloaders.euclid_transforms import (
    BANDS,
    TableNormalizer,
    load_normalizers,
    transform_table,
    validate_band_order,
)
from lib.utils.log import get_logger
from lib.utils.utils import check

logger = get_logger()

type F64 = NDArray[np.float64]
type I64 = NDArray[np.int64]
type Bool = NDArray[np.bool_]
type Arrays = dict[str, NDArray[Any]]

NZ = 601
NM = 767
N_EARLY = 8
YOUNG_EBV = 11
REGION_CELLS = (N_EARLY * NZ, (NM - N_EARLY) * NZ)
CURVES = ("SMC_Prevot", "SB_Calzetti", "SB_Calzetti_mod")
LAST_FILTER = b"Euclid/NISP.H"
NORTH_COLUMNS = (
    "CFHT/MegaCam/MegaCam.u",
    "Subaru/HSC/HSC.g",
    "CFHT/MegaCam/MegaCam.r",
    "PAN-STARRS/PS1.i",
    "Subaru/HSC/HSC.z",
    "Euclid/VIS.vis",
    "Euclid/NISP.Y",
    "Euclid/NISP.J",
    "Euclid/NISP.H",
)
SOUTH_COLUMNS = (
    "CTIO/DECam/DECam.g",
    "CTIO/DECam/DECam.r",
    "CTIO/DECam/DECam.i",
    "CTIO/DECam/DECam.z",
    "Euclid/VIS.vis",
    "Euclid/NISP.Y",
    "Euclid/NISP.J",
    "Euclid/NISP.H",
)
MODEL_NORTH = "model_flux_north.npy"
MODEL_SOUTH = "model_flux_south.npy"
GALCOEF_NORTH = "galcoef_north.npy"
GALCOEF_SOUTH = "galcoef_south.npy"
MODEL_INDEX = "model_index.parquet"

SOUTH_FIT_SLOTS = [10, 11, 12, 13, 0, 1, 2, 3]
PUBLISHED_ZPC_S = np.array([1.0, 1.0, 1.0, 1.015, 1.05, 0.95, 0.97, 0.95])
BETA_S = np.array([0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05])
EUCLID_FACTORS = ("VIS", "Y", "J", "H")
PAIRS = ((5, 10), (6, 11), (7, 12), (8, 13))
U_SLOT = 4
BPC = 1.018
SEED = 20261104
SETS = ("F_TRAIN", "F_VALID", "CDFS", "MUSE")
SOURCE_H5 = {"F_TRAIN": "dataset.h5", "F_VALID": "dataset.h5", "CDFS": "cdfs.h5", "MUSE": "ho.h5"}
Z_SOURCE = "ppub_south_zpf_median"
U_DEPTH_MAG = 24.2
U_POOL = 4000
U_BINS = 20
ST_OK, ST_FEW, ST_ZOUT, ST_NOSCALE, ST_NOLABEL = 0, 1, 2, 3, 4
STATUS_NAMES = ("OK", "FEW_BANDS", "Z_OUT_OF_GRID", "NO_POSITIVE_SCALE", "NO_VALID_LABEL")
ROW_KEYS = (
    "object_id",
    "flux",
    "error",
    "valid",
    "ancillary_values",
    "tile",
    "field_code",
    "vis_det",
)
PHOT_KEYS = (*ROW_KEYS, "ancillary_columns", "bands")


class TemplateGridConfig(BaseModel):
    north_model_fits: Path
    south_model_fits: Path
    north_galactic_grid: Path
    south_galactic_grid: Path
    out_dir: Path


class TranslationConfig(BaseModel):
    labels: Path
    f_train_photometry: Path
    f_valid_photometry: Path
    cdfs_photometry: Path
    ho_photometry: Path
    grids_dir: Path
    zero_points: Path
    z_fix_predictions: list[Path]
    z_fix_column: str = "ppub_median"
    normalizers: Path
    out_dir: Path


def interp_z(grid: F64, z: F64) -> F64:
    t = 100.0 * np.asarray(z, dtype=np.float64)
    j = np.minimum(np.floor(t).astype(np.int64), NZ - 2)
    w = t - j
    a = grid[:, j, :]
    c = grid[:, j + 1, :]
    out: F64 = np.transpose((1 - w)[None, :, None] * a + w[None, :, None] * c, (1, 0, 2))
    return out


@dataclass(frozen=True)
class Grids:
    model_north: F64
    model_south: F64
    galcoef_north: F64
    galcoef_south: F64

    @classmethod
    def load(cls, directory: Path) -> Grids:
        return cls(
            np.load(directory / MODEL_NORTH),
            np.load(directory / MODEL_SOUTH),
            np.load(directory / GALCOEF_NORTH),
            np.load(directory / GALCOEF_SOUTH),
        )

    def at_z(self, z: F64, ebv: F64) -> tuple[F64, F64, F64, F64]:
        z = np.asarray(z, dtype=np.float64)
        ebv = np.asarray(ebv, dtype=np.float64)
        ms, cs = interp_z(self.model_south, z), interp_z(self.galcoef_south, z)
        mn, cn = interp_z(self.model_north, z), interp_z(self.galcoef_north, z)
        ds = 10 ** (-0.4 * cs * BPC * ebv[:, None, None])
        dn = 10 ** (-0.4 * cn * BPC * ebv[:, None, None])
        return ms * ds, mn * dn, cs, cn


def read_grid_dat(path: Path, nb: int, last_filter: bytes = LAST_FILTER) -> F64:
    b = Path(path).read_bytes()
    occ = [m.start() for m in re.finditer(re.escape(last_filter), b)]
    blocks: list[F64] = []
    pos = 0
    for n_cells in REGION_CELLS:
        size = n_cells * nb * 16
        cands = [o for o in occ if o >= pos]
        start = None
        for i, o in enumerate(cands):
            s = o + len(last_filter)
            nxt = cands[i + 1] if i + 1 < len(cands) else len(b)
            if s + size <= nxt and s + size <= len(b):
                start = s
                break
        if start is None:
            raise ValueError(f"region block not found in {path}")
        arr = np.frombuffer(b, dtype="<f8", count=n_cells * nb * 2, offset=start)
        blocks.append(arr.reshape(n_cells, nb, 2).copy())
        pos = start + size
    return np.concatenate(blocks)


def load_model_fits(path: Path, columns: Sequence[str]) -> tuple[NDArray[Any], F64, F64]:
    with fits.open(path, memmap=False) as h:
        d = h[1].data
        sed = np.asarray(d["SED"])
        ebv = np.asarray(d["E(B-V)"])
        flux = np.stack([np.asarray(d[c], dtype=np.float64) for c in columns], axis=-1)
    return sed, ebv, flux


def model_index(sed: NDArray[Any], ebv: NDArray[Any]) -> pd.DataFrame:
    names = sed.reshape(NM, NZ)[:, 0]
    values = ebv.reshape(NM, NZ)[:, 0].astype(np.float64)
    per_sed = len(CURVES) * YOUNG_EBV
    rows: list[dict[str, object]] = []
    for m in range(NM):
        name = str(names[m]).strip().split("/")[-1]
        if m < N_EARLY:
            rows.append(
                dict(
                    m=m,
                    region="Early",
                    region_index=0,
                    sed=name,
                    sed_index=m,
                    curve="SB_Calzetti",
                    curve_index=0,
                    ebv=float(values[m]),
                    ebv_index=0,
                )
            )
        else:
            sed_i, rest = divmod(m - N_EARLY, per_sed)
            curve_i, ebv_i = divmod(rest, YOUNG_EBV)
            rows.append(
                dict(
                    m=m,
                    region="Young",
                    region_index=1,
                    sed=name,
                    sed_index=sed_i,
                    curve=CURVES[curve_i],
                    curve_index=curve_i,
                    ebv=float(values[m]),
                    ebv_index=ebv_i,
                )
            )
    return pd.DataFrame(rows)


def build_template_grids(config: TemplateGridConfig) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    sed, ebv, north = load_model_fits(config.north_model_fits, NORTH_COLUMNS)
    _, _, south = load_model_fits(config.south_model_fits, SOUTH_COLUMNS)
    galcoef_north = read_grid_dat(config.north_galactic_grid, len(NORTH_COLUMNS))[:, :, 0]
    galcoef_south = read_grid_dat(config.south_galactic_grid, len(SOUTH_COLUMNS))[:, :, 0]
    np.save(config.out_dir / MODEL_NORTH, north.reshape(NM, NZ, len(NORTH_COLUMNS)))
    np.save(config.out_dir / MODEL_SOUTH, south.reshape(NM, NZ, len(SOUTH_COLUMNS)))
    np.save(config.out_dir / GALCOEF_NORTH, galcoef_north.reshape(NM, NZ, len(NORTH_COLUMNS)))
    np.save(config.out_dir / GALCOEF_SOUTH, galcoef_south.reshape(NM, NZ, len(SOUTH_COLUMNS)))
    model_index(sed, ebv).to_parquet(config.out_dir / MODEL_INDEX, index=False)
    logger.info(f"template grids: {NM} models x {NZ} z -> {config.out_dir}")


def asinh_mag(f: F64, b: float) -> F64:
    out: F64 = 23.9 - 2.5 / np.log(10) * (np.arcsinh(f / (2 * b)) + np.log(b))
    return out


def south_inputs(flux: F64, err: F64, valid: Bool, zpc_s: F64) -> tuple[F64, F64, Bool]:
    f = flux * zpc_s
    e = err * zpc_s
    sig = np.hypot(e, BETA_S * np.abs(f))
    v = valid & np.isfinite(f) & np.isfinite(sig) & (sig > 0)
    return np.where(v, f, 0.0), np.where(v, sig, 1.0), v


def profile_fit(f: F64, sig: F64, v: Bool, m: F64) -> tuple[F64, F64, Bool]:
    w = np.where(v, 1.0 / sig**2, 0.0)
    num = np.einsum("nb,nkb->nk", np.maximum(f, 0.0) * w, m)
    den = np.einsum("nb,nkb->nk", w, m * m)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = num / den
    ok = (den > 0) & np.isfinite(alpha) & (alpha > 0)
    alpha = np.where(ok, alpha, np.nan)
    res = f[:, None, :] - np.where(ok, alpha, 0.0)[:, :, None] * m
    chi2 = np.einsum("nkb,nb->nk", res * res, w)
    chi2 = np.where(ok, chi2, np.inf)
    return alpha, chi2, ok


def fit_and_translate(
    flux14: F64,
    err14: F64,
    valid14: Bool,
    z: F64,
    ebv: F64,
    status: I64,
    grids: Grids,
    zpc_s: F64,
    chunk: int = 1024,
) -> Arrays:
    n = len(z)
    out: Arrays = dict(
        best_m=np.full(n, -1, np.int64),
        alpha=np.full(n, np.nan),
        chi2=np.full(n, np.nan),
        p_best=np.full(n, np.nan),
        n_fit_bands=np.zeros(n, np.int64),
        ratio_best=np.full((n, 4), np.nan),
        ratio_wavg=np.full((n, 4), np.nan),
        ratio_logstd=np.full((n, 4), np.nan),
        u_model_ujy=np.full(n, np.nan),
        u_wavg_ujy=np.full(n, np.nan),
        galcoef_north_best=np.full((n, 9), np.nan),
        galcoef_south_best=np.full((n, 8), np.nan),
        model_north_best=np.full((n, 9), np.nan),
        model_south_best=np.full((n, 8), np.nan),
    )
    status = status.copy()
    f, sig, v = south_inputs(
        flux14[:, SOUTH_FIT_SLOTS], err14[:, SOUTH_FIT_SLOTS], valid14[:, SOUTH_FIT_SLOTS], zpc_s
    )
    nfit = v.sum(1)
    out["n_fit_bands"] = nfit.astype(np.int64)
    few = (nfit < 3) | (v[:, :4].sum(1) < 1) | (v[:, 4:].sum(1) < 1)
    status[(status == ST_OK) & few] = ST_FEW
    todo = np.where(status == ST_OK)[0]
    for s in range(0, len(todo), chunk):
        idx = todo[s : s + chunk]
        ms, mn, cs, cn = grids.at_z(z[idx], ebv[idx])
        alpha, chi2, ok = profile_fit(f[idx], sig[idx], v[idx], ms)
        anyok = ok.any(1)
        status[idx[~anyok]] = ST_NOSCALE
        keep = np.where(anyok)[0]
        if not len(keep):
            continue
        ii = idx[keep]
        a, c2, okk = alpha[keep], chi2[keep], ok[keep]
        msk, mnk, csk, cnk = ms[keep], mn[keep], cs[keep], cn[keep]
        kb = np.argmin(c2, axis=1)
        r = np.arange(len(keep))
        c2min = c2[r, kb]
        p = np.where(okk, np.exp(-(c2 - c2min[:, None]) / 2), 0.0)
        p /= p.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = mnk[:, :, 1:5] / msk[:, :, 0:4]
        rok = np.isfinite(ratio) & (ratio > 0)
        pw = p[:, :, None] * rok
        pws = pw.sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rw = np.where(rok, ratio, 0.0)
            lr = np.where(rok, np.log(np.where(rok, ratio, 1.0)), 0.0)
            wavg = (pw * rw).sum(1) / pws
            lmean = (pw * lr).sum(1) / pws
            lstd = np.sqrt(np.maximum((pw * (lr - lmean[:, None, :]) ** 2).sum(1) / pws, 0.0))
        rb = np.where(rok[r, kb], ratio[r, kb], np.nan)
        out["best_m"][ii] = kb
        out["alpha"][ii] = a[r, kb]
        out["chi2"][ii] = c2min
        out["p_best"][ii] = p[r, kb]
        out["ratio_best"][ii] = rb
        out["ratio_wavg"][ii] = wavg
        out["ratio_logstd"][ii] = lstd
        out["u_model_ujy"][ii] = np.maximum(0.0, a[r, kb] * mnk[r, kb, 0])
        out["u_wavg_ujy"][ii] = np.nansum(p * np.where(okk, a, 0.0) * mnk[:, :, 0], axis=1)
        out["galcoef_north_best"][ii] = cnk[r, kb]
        out["galcoef_south_best"][ii] = csk[r, kb]
        out["model_north_best"][ii] = mnk[r, kb]
        out["model_south_best"][ii] = msk[r, kb]
    out["status"] = status
    with np.errstate(divide="ignore", invalid="ignore"):
        out["poor_fit"] = (status == ST_OK) & (out["chi2"] / np.maximum(nfit - 1, 1) > 10)
    return out


def assemble(
    flux14: F64,
    err14: F64,
    valid14: Bool,
    fit: Mapping[str, NDArray[Any]],
    u_obs: F64,
    sigma_u: F64,
    kzpc: F64,
) -> tuple[F64, F64, Bool]:
    n = len(flux14)
    ratio = fit["ratio_best"]
    ok = fit["status"] == ST_OK
    f = np.full((n, len(BANDS)), np.nan)
    e = np.full((n, len(BANDS)), np.nan)
    f[:, 0:4] = flux14[:, 0:4]
    e[:, 0:4] = err14[:, 0:4]
    for j, (ns, ss) in enumerate(PAIRS):
        rv = ratio[:, j]
        vv = valid14[:, ss] & ok & np.isfinite(rv) & (rv > 0)
        f[vv, ns] = kzpc[j] * rv[vv] * flux14[vv, ss]
        e[vv, ns] = kzpc[j] * rv[vv] * err14[vv, ss]
    uv = ok & np.isfinite(u_obs) & np.isfinite(sigma_u) & (sigma_u > 0)
    f[uv, U_SLOT] = u_obs[uv]
    e[uv, U_SLOT] = sigma_u[uv]
    mask = np.isfinite(f) & np.isfinite(e) & (e > 0)
    mask[:, 0:4] = valid14[:, 0:4]
    f[:, 5:9] = np.where(mask[:, 5:9], f[:, 5:9], np.nan)
    e[:, 5:9] = np.where(mask[:, 5:9], e[:, 5:9], np.nan)
    return f, e, mask


class SigmaU:
    def __init__(self, flux_u: F64, err_u: F64) -> None:
        self.b_u = float(np.median(err_u))
        m = asinh_mag(flux_u, self.b_u)
        self.edges: F64 = np.quantile(m, np.linspace(0, 1, U_BINS + 1))
        pool_bins = np.clip(np.digitize(m, self.edges[1:-1]), 0, U_BINS - 1)
        self.members = [np.sort(err_u[pool_bins == b]) for b in range(U_BINS)]
        self.counts = np.array([len(x) for x in self.members])

    def bins(self, u_model: F64) -> I64:
        m = asinh_mag(u_model, self.b_u)
        out: I64 = np.clip(np.digitize(m, self.edges[1:-1]), 0, U_BINS - 1)
        return out

    def draw(self, u_model: F64, rng: np.random.Generator) -> tuple[F64, F64, I64]:
        n = len(u_model)
        b = self.bins(np.where(np.isfinite(u_model), u_model, 0.0))
        u01 = rng.random(n)
        eps = rng.standard_normal(n)
        sig = np.array(
            [
                self.members[bb][min(int(x * self.counts[bb]), self.counts[bb] - 1)]
                for bb, x in zip(b, u01, strict=True)
            ]
        )
        return sig, eps, b


def depth_noise_model() -> SigmaU:
    rng = np.random.default_rng(0)
    flux = rng.uniform(0.1, 10.0, U_POOL)
    sigma = 10 ** ((23.9 - U_DEPTH_MAG) / 2.5) / 5
    return SigmaU(flux, np.full(U_POOL, sigma))


def load_photometry(path: Path) -> Arrays:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in PHOT_KEYS}
    validate_band_order(d["bands"])
    check(d["ancillary_columns"][0] == "GAL_EBV", "GAL_EBV is the first ancillary column")
    expected = np.isfinite(d["flux"]) & np.isfinite(d["error"]) & (d["error"] > 0)
    check(np.array_equal(d["valid"], expected), "valid mask")
    return d


def align_rows(d: Mapping[str, NDArray[Any]], ids: I64) -> Arrays:
    pos = {int(o): i for i, o in enumerate(d["object_id"])}
    j = np.array([pos[int(o)] for o in ids])
    out = {k: (v[j] if k in ROW_KEYS else v) for k, v in d.items()}
    check(
        np.array_equal(out["object_id"], ids) and bool((out["field_code"] == 1).all()),
        "alignment",
    )
    return out


def impute_ebv(values: F64, tile: I64) -> tuple[F64, Bool]:
    ebv = values.copy()
    imputed = ~np.isfinite(ebv) | (ebv < 0)
    for i in np.where(imputed)[0]:
        same = (tile == tile[i]) & ~imputed
        ebv[i] = np.median(values[same]) if same.any() else np.median(values[~imputed])
    return ebv, imputed


def load_labels(path: Path) -> Arrays:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_euclid_factors(path: Path) -> dict[str, float]:
    factors = json.loads(Path(path).read_text())["factors"]["euclid"]
    return {name: float(factors[name]) for name in EUCLID_FACTORS}


def south_zero_points(euclid: Mapping[str, float]) -> F64:
    fitted = np.array([euclid[name] for name in EUCLID_FACTORS], dtype=np.float64)
    return np.concatenate((PUBLISHED_ZPC_S[:4], fitted))


def label_free_z(paths: Sequence[Path], column: str) -> pd.Series:
    frames = [pd.read_parquet(p, columns=["object_id", column]) for p in paths]
    return pd.concat(frames).drop_duplicates("object_id").set_index("object_id")[column]


def label_free_z_for(series: pd.Series, ids: I64) -> F64:
    z = series.reindex(ids).to_numpy(np.float64)
    out: F64 = np.where(np.isfinite(z) & (z >= 0), z, np.nan)
    return out


def translate_set(
    name: str,
    d: Mapping[str, NDArray[Any]],
    labels: Mapping[str, NDArray[Any]],
    z: F64,
    grids: Grids,
    index: pd.DataFrame,
    zpc_s: F64,
    noise: SigmaU,
    table: TableNormalizer,
    rng: np.random.Generator,
) -> Arrays:
    n = len(d["object_id"])
    status = np.zeros(n, np.int64)
    status[~np.isfinite(z)] = ST_NOLABEL
    if name == "MUSE":
        rs = labels[f"{name}__reference_status"]
        outside = labels[f"{name}__reference_outside_grid"]
        status[(status == 0) & (rs != "valid_original_reference")] = ST_NOLABEL
        status[(status == 0) & outside] = ST_ZOUT
    status[(status == 0) & ((z < 0) | (z > 6))] = ST_ZOUT
    ebv, imputed = impute_ebv(d["ancillary_values"][:, 0], d["tile"])
    fit = fit_and_translate(
        d["flux"],
        d["error"],
        d["valid"],
        np.where(status == 0, z, np.nan),
        ebv,
        status,
        grids,
        zpc_s,
    )
    sig_u, eps, ubin = noise.draw(fit["u_model_ujy"], rng)
    ok = fit["status"] == ST_OK
    u_obs = np.where(ok, fit["u_model_ujy"] + sig_u * eps, np.nan)
    kzpc = zpc_s[:4].copy()
    f14, e14, m14 = assemble(
        d["flux"], d["error"], d["valid"], fit, u_obs, np.where(ok, sig_u, np.nan), kzpc
    )
    tv = transform_table(f14, e14, m14, table)
    bm = fit["best_m"]
    known = np.maximum(bm, 0)
    return dict(
        object_id=d["object_id"],
        set=np.array(name),
        source_h5=np.array(SOURCE_H5[name]),
        source_row=labels[f"{name}__source_row"],
        bands=np.array(BANDS),
        flux_ujy=f14,
        error_ujy=e14,
        table_mask=m14,
        table_values=tv,
        table_available=m14.sum(1) >= 2,
        z_fix=z,
        z_source=np.array([Z_SOURCE] * n),
        label_tier=labels[f"{name}__label_tier"],
        label_source_code=labels[f"{name}__label_source_code"],
        label_kind=labels[f"{name}__kind"],
        label_family=labels[f"{name}__family"],
        reference_status=labels[f"{name}__reference_status"],
        status=fit["status"].astype(np.uint8),
        ebv_imputed=imputed,
        poor_fit=fit["poor_fit"],
        gal_ebv=ebv,
        best_m=bm,
        best_sed=np.where(bm >= 0, index.sed.to_numpy()[known], "").astype(str),
        best_curve=np.where(bm >= 0, index.curve.to_numpy()[known], "").astype(str),
        best_ebv=np.where(bm >= 0, index.ebv.to_numpy()[known], np.nan),
        alpha=fit["alpha"],
        chi2=fit["chi2"],
        n_fit_bands=fit["n_fit_bands"],
        p_best=fit["p_best"],
        ratio_best=fit["ratio_best"],
        ratio_wavg=fit["ratio_wavg"],
        ratio_logstd=fit["ratio_logstd"],
        k_zpc=kzpc,
        u_model_ujy=fit["u_model_ujy"],
        u_wavg_ujy=fit["u_wavg_ujy"],
        sigma_u_ujy=np.where(ok, sig_u, np.nan),
        u_eps=np.where(ok, eps, np.nan),
        u_sigma_bin=np.where(ok, ubin, -1),
        galcoef_north_best=fit["galcoef_north_best"],
        galcoef_south_best=fit["galcoef_south_best"],
        model_north_best=fit["model_north_best"],
        model_south_best=fit["model_south_best"],
        src_flux_ujy=d["flux"],
        src_error_ujy=d["error"],
        src_valid=d["valid"],
        vis_det=d["vis_det"],
        ancillary_values=d["ancillary_values"],
        ancillary_columns=d["ancillary_columns"],
        tile=d["tile"],
    )


def run_translation(config: TranslationConfig) -> list[Path]:
    grids = Grids.load(config.grids_dir)
    index = pd.read_parquet(config.grids_dir / MODEL_INDEX)
    labels = load_labels(config.labels)
    zpc_s = south_zero_points(load_euclid_factors(config.zero_points))
    table = load_normalizers(config.normalizers).table
    z_free = label_free_z(config.z_fix_predictions, config.z_fix_column)
    sources = {
        "F_TRAIN": config.f_train_photometry,
        "F_VALID": config.f_valid_photometry,
        "CDFS": config.cdfs_photometry,
        "MUSE": config.ho_photometry,
    }
    noise = depth_noise_model()
    children = np.random.SeedSequence(SEED).spawn(len(SETS))
    config.out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"south zero points {zpc_s.tolist()}")
    written: list[Path] = []
    for si, name in enumerate(SETS):
        ids = labels[f"{name}__object_id"]
        d = align_rows(load_photometry(sources[name]), ids)
        z = label_free_z_for(z_free, ids)
        rng = np.random.default_rng(children[si])
        res = translate_set(name, d, labels, z, grids, index, zpc_s, noise, table, rng)
        path = config.out_dir / f"translated_{name}.npz"
        np.savez(path, **cast(dict[str, Any], res))
        written.append(path)
        counts = {
            STATUS_NAMES[k]: int((res["status"] == k).sum()) for k in range(len(STATUS_NAMES))
        }
        logger.info(
            f"{name}: rows={len(ids)} status={counts} poor_fit={int(res['poor_fit'].sum())} "
            f"table_available={int(res['table_available'].sum())} -> {path}"
        )
    return written
