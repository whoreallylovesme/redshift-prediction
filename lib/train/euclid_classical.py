from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import pandas as pd
import pygam
import torch
from catboost import CatBoostRegressor
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from lib.dataloaders.euclid_packet import select_rows
from lib.metrics.bootstrap import photoz_stats
from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as

logger = get_logger()

type Learner = Literal["catboost", "rf", "annz", "llr", "gam"]
type Floats = NDArray[np.float64]
type Table = tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.bool_]]
type Features = tuple[Floats, Floats]

SEED = 20261027
SLOTS = 9
GAM_CONTINUOUS = 2 * SLOTS
GAM_SPLINES = 10
GAM_ORDER = 3
GAM_LAMBDAS = np.logspace(-5, 5, 11)
GAM_CHUNK = 4096
LLR_NEIGHBOURS = 1000
LLR_CHUNK = 512
ANNZ_HIDDEN = 10
ANNZ_COMMITTEE = 5
ANNZ_STEPS = 150
ANNZ_VALID = 0.2
VALID_REFERENCE = "valid_original_reference"
PARAMS: dict[str, dict[str, object]] = {
    "catboost": {
        "iterations": 800,
        "depth": 6,
        "learning_rate": 0.05,
        "l2_leaf_reg": 5.0,
        "border_count": 128,
        "bootstrap_type": "No",
        "random_strength": 0.0,
        "loss_function": "RMSE",
    },
    "rf": {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    "llr": {"k": LLR_NEIGHBOURS, "metric": "euclidean", "local_fit": "OLS with intercept"},
    "annz": {
        "hidden": [ANNZ_HIDDEN, ANNZ_HIDDEN],
        "activation": "tanh",
        "optimizer": "LBFGS strong_wolfe",
        "committee": ANNZ_COMMITTEE,
        "internal_validation": ANNZ_VALID,
    },
    "gam": {
        "continuous": GAM_CONTINUOUS,
        "n_splines": GAM_SPLINES,
        "spline_order": GAM_ORDER,
        "lam_grid": GAM_LAMBDAS.tolist(),
        "selection": "GCV",
    },
}
TARGETS: dict[str, str] = {
    "catboost": "log1p(z)",
    "rf": "z",
    "llr": "z",
    "annz": "z",
    "gam": "z",
}


class EuclidClassicalConfig(BaseModel):
    learner: Learner
    out_dir: Path
    pilot_packet: Path
    cdfs_packet: Path
    cdfs_labels: Path
    ho_packet: Path
    ho_native: Path
    ho_reference: Path
    translated_f_train: Path
    translated_f_valid: Path
    translated_cdfs: Path
    translated_muse: Path
    split_meta: Path
    predict_sets: dict[str, Path]
    threads: int = 8


@dataclass(frozen=True)
class TrainingRows:
    x: Floats
    y: Floats
    groups: NDArray[np.int64]
    sets: dict[str, tuple[NDArray[np.int64], Floats]]
    rows: dict[str, dict[str, int] | int]
    valid_z: Floats


def override(
    ids: NDArray[np.int64], tv: NDArray[np.float32], tm: NDArray[np.bool_], path: Path
) -> tuple[NDArray[np.float32], NDArray[np.bool_], int]:
    d = np.load(path)
    pos = {int(o): j for j, o in enumerate(d["object_id"])}
    status, values, mask = d["status"], d["table_values"], d["table_mask"]
    tv, tm = tv.copy(), tm.copy()
    hit = 0
    for i, o in enumerate(ids):
        j = pos.get(int(o), -1)
        if j >= 0 and int(status[j]) == 0:
            tv[i], tm[i] = values[j], mask[j]
            hit += 1
    return tv, tm, hit


def features(tv: NDArray[np.float32], tm: NDArray[np.bool_]) -> Features:
    v = tv[:, :SLOTS, :].astype(np.float64)
    m = tm[:, :SLOTS].astype(bool)
    x = np.concatenate([np.where(m, v[:, :, 0], np.nan), np.where(m, v[:, :, 1], np.nan)], 1)
    return x, m.astype(np.float64)


def packet_rows(
    path: Path,
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.bool_], NDArray[np.int64]]:
    with h5py.File(path, "r") as h:
        return h["object_id"][:], h["table_values"][:], h["table_mask"][:], h["owner_tile"][:]


def ho_labels(
    cfg: EuclidClassicalConfig, hid: NDArray[np.int64]
) -> tuple[NDArray[np.integer], Floats, NDArray[np.bool_]]:
    nat = np.load(cfg.ho_native, allow_pickle=False)
    pos = {int(o): i for i, o in enumerate(nat["object_id"])}
    field = nat["field_code"][np.array([pos[int(o)] for o in hid])]
    ref = (
        pd.read_parquet(cfg.ho_reference)
        .drop_duplicates("object_id")
        .set_index("object_id")
        .loc[hid]
    )
    hz = ref.z_reference.to_numpy(float)
    ok = (
        (ref.reference_status.to_numpy() == VALID_REFERENCE)
        & np.isfinite(hz)
        & ~ref.reference_outside_model_grid.to_numpy(bool)
    )
    return field, hz, ok


def training_rows(cfg: EuclidClassicalConfig) -> TrainingRows:
    rep: dict[str, dict[str, int] | int] = {}
    with h5py.File(cfg.pilot_packet, "r") as h:
        ti, vi, ni = h["train_indices"][:], h["valid_indices"][:], h["test_indices"][:]
        allids, owner = h["object_id"][:], h["owner_tile"][:]
        tv_all, tm_all = h["table_values"], h["table_mask"]
        train: Table = (allids[ti], select_rows(tv_all, ti), select_rows(tm_all, ti))
        valid: Table = (allids[vi], select_rows(tv_all, vi), select_rows(tm_all, vi))
        north: Table = (allids[ni], select_rows(tv_all, ni), select_rows(tm_all, ni))
        zt, zv = h["train_z"][:].astype(float), h["valid_z"][:].astype(float)
    fvalid_tiles = np.unique(owner[vi])
    known = np.r_[allids[ti], allids[vi]]
    tv_t, tm_t, hit = override(*train, cfg.translated_f_train)
    rep["F_TRAIN"] = {"n": len(train[0]), "translated": hit}
    xs, ys, gs, ids = [features(tv_t, tm_t)], [zt], [owner[ti]], [train[0]]
    cid, ctv, ctm, ctile = packet_rows(cfg.cdfs_packet)
    lab = pd.read_parquet(cfg.cdfs_labels).drop_duplicates("object_id").set_index("object_id")
    keep = np.isin(cid, lab.index.values)
    cz = np.full(len(cid), np.nan)
    cz[keep] = lab.loc[cid[keep]].z.to_numpy(float)
    keep &= np.isfinite(cz) & (cz >= 0) & ~np.isin(ctile, fvalid_tiles) & ~np.isin(cid, known)
    known = np.r_[known, cid[keep]]
    a, b, hit = override(cid[keep], ctv[keep], ctm[keep], cfg.translated_cdfs)
    xs.append(features(a, b))
    ys.append(cz[keep])
    gs.append(ctile[keep])
    ids.append(cid[keep])
    rep["CDFS"] = {"n": int(keep.sum()), "translated": hit}
    hid, htv, htm, htile = packet_rows(cfg.ho_packet)
    field, hz, ok_h = ho_labels(cfg, hid)
    for fcode, name in ((1, "MUSE"), (2, "HO_S")):
        keep = (
            ok_h
            & (field == fcode)
            & (hz >= 0)
            & ~np.isin(htile, fvalid_tiles)
            & ~np.isin(hid, known)
        )
        known = np.r_[known, hid[keep]]
        if fcode == 1:
            a, b, hit = override(hid[keep], htv[keep], htm[keep], cfg.translated_muse)
        else:
            a, b, hit = htv[keep], htm[keep], 0
        xs.append(features(a, b))
        ys.append(hz[keep])
        gs.append(htile[keep])
        ids.append(hid[keep])
        rep[name] = {"n": int(keep.sum()), "translated": hit}
    vtv, vtm, hit = override(*valid, cfg.translated_f_valid)
    rep["F_VALID"] = {"n": len(valid[0]), "translated": hit}
    xs.append(features(vtv, vtm))
    ys.append(zv)
    gs.append(owner[vi])
    ids.append(valid[0])
    xc = np.concatenate([x for x, _ in xs])
    mc = np.concatenate([m for _, m in xs])
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    ids_all = np.concatenate(ids)
    zval = np.asarray(json.loads(cfg.split_meta.read_text())["zval"]["ids"], np.int64)
    check(len(np.unique(ids_all)) == len(ids_all), "labelled rows must be unique objects")
    check(np.isin(zval, ids_all).all(), "validation ids must be labelled rows")
    sel = ~np.isin(ids_all, zval)
    xc, mc, y, g = xc[sel], mc[sel], y[sel], g[sel]
    med = np.nanmedian(xc, 0)
    rep["n_train"] = len(y)
    raw: dict[str, tuple[NDArray[np.int64], Features]] = {
        "valid": (valid[0], features(vtv, vtm)),
        "north": (north[0], features(north[1], north[2])),
    }
    for name, path in cfg.predict_sets.items():
        pid, ptv, ptm, _ = packet_rows(path)
        raw[name] = (pid, features(ptv, ptm))

    def matrix(xm: Features) -> Floats:
        x, m = xm
        out: Floats = np.c_[np.where(np.isfinite(x), x, med[None]), m]
        return out

    return TrainingRows(
        x=matrix((xc, mc)),
        y=y,
        groups=g,
        sets={k: (v[0], matrix(v[1])) for k, v in raw.items()},
        rows=rep,
        valid_z=zv,
    )


def standardize(xt: Floats, *others: Floats) -> list[Floats]:
    mu, sd = xt.mean(0), xt.std(0)
    sd[sd == 0] = 1.0
    return [(a - mu) / sd for a in (xt, *others)]


def fit_llr(z: Floats, y: Floats, q: Floats) -> Floats:
    nn = NearestNeighbors(n_neighbors=LLR_NEIGHBOURS, algorithm="brute").fit(z)
    out = np.empty(len(q))
    for start in range(0, len(q), LLR_CHUNK):
        idx = nn.kneighbors(q[start : start + LLR_CHUNK], return_distance=False)
        for o, ii in enumerate(idx):
            a = np.c_[np.ones(len(ii)), z[ii]]
            coef, *_ = np.linalg.lstsq(a, y[ii], rcond=None)
            out[start + o] = np.r_[1.0, q[start + o]] @ coef
    return out


def fit_annz(z: Floats, y: Floats, groups: NDArray[np.int64], q: Floats, threads: int) -> Floats:
    torch.set_num_threads(threads)
    splitter = GroupShuffleSplit(n_splits=1, test_size=ANNZ_VALID, random_state=SEED)
    ((tr, va),) = splitter.split(z, y, groups)
    xt, yt = torch.tensor(z[tr]), torch.tensor(y[tr])
    xa, ya = torch.tensor(z[va]), torch.tensor(y[va])
    xq = torch.tensor(q)
    preds = []
    for seed in range(ANNZ_COMMITTEE):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(z.shape[1], ANNZ_HIDDEN),
            torch.nn.Tanh(),
            torch.nn.Linear(ANNZ_HIDDEN, ANNZ_HIDDEN),
            torch.nn.Tanh(),
            torch.nn.Linear(ANNZ_HIDDEN, 1),
        ).double()
        opt = torch.optim.LBFGS(
            net.parameters(),
            lr=1.0,
            max_iter=20,
            history_size=20,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-14,
        )
        closure = annz_closure(net, opt, xt, yt)
        best, state, prev = np.inf, None, 0
        for _ in range(ANNZ_STEPS):
            opt.step(closure)
            with torch.no_grad():
                vl = float(((net(xa).squeeze(1) - ya) ** 2).mean())
            if vl < best:
                best, state = vl, {k: v.clone() for k, v in net.state_dict().items()}
            it = int(opt.state[opt.param_groups[0]["params"][0]]["n_iter"])
            if it == prev:
                break
            prev = it
        if state is None:
            raise ValueError("ANNz validation loss must be finite")
        net.load_state_dict(state)
        with torch.no_grad():
            preds.append(net(xq).squeeze(1).numpy().copy())
    out: Floats = np.mean(preds, 0)
    return out


def annz_closure(
    net: torch.nn.Module, opt: torch.optim.Optimizer, xt: torch.Tensor, yt: torch.Tensor
) -> Callable[[], torch.Tensor]:
    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss: torch.Tensor = ((net(xt).squeeze(1) - yt) ** 2).mean()
        loss.backward()
        return loss

    return closure


def fit_gam(z: Floats, y: Floats, q: Floats) -> tuple[Floats, float]:
    terms = pygam.s(0, n_splines=GAM_SPLINES, spline_order=GAM_ORDER)
    for j in range(1, GAM_CONTINUOUS):
        terms += pygam.s(j, n_splines=GAM_SPLINES, spline_order=GAM_ORDER)
    for j in range(GAM_CONTINUOUS, z.shape[1]):
        terms += pygam.l(j)
    gam = pygam.LinearGAM(terms, max_iter=100, tol=1e-4, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gam.gridsearch(z, y, objective="GCV", progress=False, lam=GAM_LAMBDAS, keep_best=True)
    pred = np.concatenate([gam.predict(q[i : i + GAM_CHUNK]) for i in range(0, len(q), GAM_CHUNK)])
    return pred, float(np.ravel(gam.lam[0])[0])


def fit_predict(
    learner: Learner,
    xt: Floats,
    yt: Floats,
    groups: NDArray[np.int64],
    q: Floats,
    threads: int,
) -> tuple[Floats, dict[str, float]]:
    if learner == "catboost":
        model = CatBoostRegressor(
            **PARAMS["catboost"],
            random_seed=SEED,
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
        ).fit(xt, np.log1p(yt))
        return np.expm1(model.predict(q)), {}
    if learner == "rf":
        forest = RandomForestRegressor(**PARAMS["rf"], random_state=SEED, n_jobs=threads)
        return forest.fit(xt, yt).predict(q), {}
    zt, zq = standardize(xt, q)
    if learner == "llr":
        return fit_llr(zt, yt, zq), {}
    if learner == "annz":
        gg = np.where(groups < 0, -np.arange(len(groups)) - 2, groups)
        return fit_annz(zt, yt, gg, zq, threads), {}
    pred, lam = fit_gam(zt, yt, zq)
    return pred, {"lam": lam}


def run_train(config_path: Path) -> None:
    cfg = load_yaml_as(str(config_path), EuclidClassicalConfig)
    start = time.monotonic()
    with threadpool_limits(limits=cfg.threads):
        data = training_rows(cfg)
        names = list(data.sets)
        query = np.vstack([data.sets[k][1] for k in names])
        q, log = fit_predict(cfg.learner, data.x, data.y, data.groups, query, cfg.threads)
    q = np.clip(q, 0, None)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    res: dict[str, Any] = {}
    preds: dict[str, Floats] = {}
    offset = 0
    for k in names:
        n = len(data.sets[k][0])
        preds[k] = q[offset : offset + n]
        res[f"{k}_object_id"], res[f"{k}_z"] = data.sets[k][0], preds[k]
        offset += n
    np.savez(cfg.out_dir / "predictions.npz", **res)
    fvalid_nmad = photoz_stats(preds["valid"], data.valid_z)[0]
    done = {
        "learner": cfg.learner,
        "params": PARAMS[cfg.learner],
        "target": TARGETS[cfg.learner],
        "seed": SEED,
        "rows": data.rows,
        "n_features": int(data.x.shape[1]),
        "fvalid_nmad": fvalid_nmad,
        "log": log,
        "seconds": time.monotonic() - start,
    }
    (cfg.out_dir / "complete.json").write_text(json.dumps(done, indent=1, default=float) + "\n")
    logger.info(
        "classical %s: n_train=%d, F-valid NMAD %.4f, %.0f s",
        cfg.learner,
        data.rows["n_train"],
        fvalid_nmad,
        done["seconds"],
    )
