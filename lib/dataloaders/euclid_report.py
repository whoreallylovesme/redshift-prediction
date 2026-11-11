from __future__ import annotations

import collections
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pydantic import BaseModel

from lib.utils.utils import check

VALID_REFERENCE = "valid_original_reference"
NORTH_SETS = ("north", "nb")
NORTH_SOURCE = "DESI"
HETDEX_SOURCE = "HETDEX"
HETDEX_SET = "h6"
VALID_SET = "valid"
HO_SET = "ho"
ZVAL_SET = "zval"
PRIMARY_STRATUM = "oii"
FIELD_F, FIELD_S = 1, 2
HO_BLOCKS = (("HO_S", FIELD_S), ("HO_F_MUSE", FIELD_F))
CLASSICAL_SETS = ("valid", "north", "nb", "d2", "d3", "d4", "d5", "h6")
N_SUBSETS = ("N-A", "N-B")
N_VIS_BINS = ((0, 21), (21, 22.5), (22.5, 30))
N_Z_BINS = ((0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 9))
MIN_SUBSET = 30
MAG_ZERO = 23.9
MIN_FLUX = 1e-6
WIDTH_FLOOR = 0.005
WIDTH_DEFAULT = 0.05
MODEL_ARGS = ("bins", "z_reweight", "head", "bands", "epochs", "seed")

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type Sample = tuple[FloatArray, FloatArray]
type SetPair = tuple[FloatArray, FloatArray]


class LabelSet(BaseModel):
    labels: Path
    column: str
    packet: Path
    phz: Path
    source: str


class PhosphorosSource(BaseModel):
    files: list[Path]
    hetdex: Path


class ReportConfig(BaseModel):
    tag: str = "P_final"
    run_name: str = "final"
    seeds: list[int]
    meta_dir: Path
    pred_dir: Path
    clip_dir: Path
    clip_seeds: list[int]
    classical_dir: Path
    learners: list[str]
    out_dir: Path
    packet: Path
    phz_lookup: Path
    cdfs_packet: Path
    cdfs_labels: Path
    cdfs_phz: Path
    ho_packet: Path
    ho_photometry: Path
    ho_reference: Path
    ho_phz: Path
    north_labels: Path
    extra_sets: dict[str, LabelSet]
    hetdex_packet: Path
    hetdex_labels: Path
    hetdex_meta: Path
    hetdex_phz: Path
    phosphoros: PhosphorosSource
    phosphoros_zpf: PhosphorosSource
    store_rows: Path


@dataclass
class SeedStack:
    ids: IntArray
    z: FloatArray
    s: FloatArray

    @property
    def mean(self) -> FloatArray:
        out: FloatArray = self.z.mean(0)
        return out


@dataclass
class EvalBlock:
    name: str
    sets: tuple[str, ...]
    rows: IntArray
    z: FloatArray
    tiles: NDArray[Any]
    phz: FloatArray


def run_names(tag: str, seeds: Sequence[int]) -> list[str]:
    return [f"{tag}_{seed}" for seed in sorted(seeds)]


def load_predictions(pred_dir: Path, tag: str, seeds: Sequence[int]) -> dict[str, SeedStack]:
    check(len(seeds) > 0, f"no runs for {tag}")
    ids: dict[str, IntArray] = {}
    zs: dict[str, list[FloatArray]] = {}
    ss: dict[str, list[FloatArray]] = {}
    for name in run_names(tag, seeds):
        with np.load(pred_dir / f"{name}_pred.npz") as p:
            for key in p.files:
                if not key.endswith("_mu"):
                    continue
                set_name = key[:-3]
                set_ids = p[f"{set_name}_object_id"]
                ids.setdefault(set_name, set_ids)
                check(np.array_equal(ids[set_name], set_ids), f"aligned ids in {set_name}")
                zs.setdefault(set_name, []).append(np.expm1(p[f"{set_name}_base"] + p[key]))
                ss.setdefault(set_name, []).append(p[f"{set_name}_s"])
    return {
        k: SeedStack(ids=ids[k], z=np.stack(zs[k]), s=np.stack(ss[k])) for k in ids
    }


def load_metas(meta_dir: Path, tag: str, seeds: Sequence[int]) -> dict[str, dict[str, Any]]:
    return {
        name: json.loads((meta_dir / f"{name}.json").read_text())
        for name in run_names(tag, seeds)
    }


def by_name(files: Sequence[Path]) -> list[Path]:
    return sorted(files, key=lambda f: f.name)


def phosphoros_series(files: Sequence[Path]) -> pd.Series:
    t = pd.concat(
        [pd.read_parquet(f, columns=["object_id", "ppub_median"]) for f in by_name(files)]
    )
    s = t.drop_duplicates("object_id").set_index("object_id").ppub_median
    return s.where(np.isfinite(s) & (s >= 0))


def phosphoros_points(files: Sequence[Path]) -> tuple[dict[int, float], dict[int, float]]:
    points: dict[int, float] = {}
    widths: dict[int, float] = {}
    for f in by_name(files):
        t = pd.read_parquet(f)
        lo, hi = t.ppub_lo68.to_numpy(float), t.ppub_hi68.to_numpy(float)
        ids, med = t.object_id.to_numpy(np.int64), t.ppub_median.to_numpy(float)
        for o, v, lo_, hi_ in zip(ids, med, lo, hi, strict=True):
            if np.isfinite(v) and v >= 0 and int(o) not in points:
                points[int(o)] = v
                widths[int(o)] = (
                    max((np.log1p(hi_) - np.log1p(max(lo_, 0))) / 2, WIDTH_FLOOR)
                    if np.isfinite(lo_) and np.isfinite(hi_)
                    else np.nan
                )
    return points, widths


def phosphoros_column(path: Path, ids: IntArray) -> tuple[FloatArray, pd.DataFrame]:
    t = pd.read_parquet(path).drop_duplicates("object_id").set_index("object_id").reindex(ids)
    m = t.ppub_median.to_numpy(float)
    return np.where(np.isfinite(m) & (m >= 0), m, np.nan), t


def phosphoros_width(lo: FloatArray, hi: FloatArray) -> FloatArray:
    out: FloatArray = np.where(
        np.isfinite(lo) & np.isfinite(hi),
        np.maximum((np.log1p(hi) - np.log1p(np.maximum(lo, 0))) / 2, WIDTH_FLOOR),
        np.nan,
    )
    return out


def phz_map(paths: Sequence[Path]) -> dict[int, float]:
    out: dict[int, float] = {}
    for path in paths:
        with np.load(path) as d:
            ids, median = d["object_id"].tolist(), d["PHZ_MEDIAN"].reshape(-1).tolist()
        out.update(zip(ids, median, strict=True))
    return out


def phz_values(paths: Sequence[Path], ids: IntArray) -> FloatArray:
    phz = phz_map(paths)
    v = np.array([phz.get(int(o), np.nan) for o in ids], float)
    out: FloatArray = np.where(v >= 0, v, np.nan)
    return out


def north_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path).drop_duplicates("object_id").set_index("object_id")


def packet_tiles(path: Path) -> tuple[IntArray, NDArray[Any]]:
    with h5py.File(path, "r") as h:
        return h["object_id"][:], h["owner_tile"][:]


def classical_series(path: Path) -> dict[str, pd.Series]:
    with np.load(path) as p:
        return {
            nm: pd.Series(p[f"{nm}_z"], index=p[f"{nm}_object_id"]) for nm in CLASSICAL_SETS
        }


def north_frame(
    preds: Mapping[str, SeedStack], nt: pd.DataFrame, pub: pd.Series, zpf: pd.Series
) -> pd.DataFrame:
    ids = np.concatenate([preds[s].ids for s in NORTH_SETS])
    zml = np.concatenate([preds[s].mean for s in NORTH_SETS])
    keep = np.isin(ids, nt.index.values)
    ids, zml = ids[keep], zml[keep]
    t = nt.loc[ids]
    phz = t.phz.to_numpy(float)
    d = pd.DataFrame(
        {
            "z": t.z_strong.to_numpy(float),
            "vis": t.vis.to_numpy(float),
            "ml": zml,
            "ppub": pub.reindex(ids).to_numpy(float),
            "zpf": zpf.reindex(ids).to_numpy(float),
            "poff": np.where(phz >= 0, phz, np.nan),
        },
        index=ids,
    )
    return d[np.isfinite(d[["z", "ml", "ppub", "zpf", "poff"]]).all(axis=1)]


def hetdex_frame(cfg: ReportConfig, h6: SeedStack) -> pd.DataFrame:
    ids = h6.ids
    lab = pd.read_parquet(cfg.hetdex_labels).set_index("object_id").reindex(ids)
    meta = pd.read_parquet(cfg.hetdex_meta).set_index("object_id").reindex(ids)
    pub = (
        pd.read_parquet(cfg.phosphoros.hetdex)
        .drop_duplicates("object_id")
        .set_index("object_id")
        .ppub_median.reindex(ids)
    )
    d = pd.DataFrame(
        {
            "z": lab.z.to_numpy(float),
            "ml": h6.mean,
            "ppub": pub.where(pub >= 0).to_numpy(float),
            "poff": phz_values([cfg.hetdex_phz], ids),
            "stratum": meta.stratum.to_numpy(),
            "vis": meta.vis.to_numpy(float),
        },
        index=ids,
    )
    keep = (
        (d.stratum == PRIMARY_STRATUM)
        & np.isfinite(d[["z", "ml", "ppub", "poff"]]).all(axis=1)
        & (d.z > 0)
    )
    return d[keep]


def extra_labels(spec: LabelSet) -> tuple[pd.DataFrame, IntArray, NDArray[Any]]:
    lt = pd.read_parquet(spec.labels).drop_duplicates("object_id").set_index("object_id")
    ids, tiles = packet_tiles(spec.packet)
    keep = np.isin(ids, lt.index.values)
    return lt, ids[keep], tiles[keep]


def sources_frame(
    cfg: ReportConfig,
    n: pd.DataFrame,
    hx: pd.DataFrame,
    preds: Mapping[str, SeedStack],
    nt: pd.DataFrame,
    pub: pd.Series,
    zpf: pd.Series,
    classical: Mapping[str, Mapping[str, pd.Series]],
) -> pd.DataFrame:
    keys = list(classical)
    north_cl = {k: pd.concat([classical[k][s] for s in NORTH_SETS]) for k in keys}
    frames = [
        pd.DataFrame(
            {
                "z": n.z.values,
                "ml": n.ml.values,
                "ppub": n.ppub.values,
                "zpf": n.zpf.values,
                "poff": n.poff.values,
                "tile": nt.tile.reindex(n.index).to_numpy(),
                "src": NORTH_SOURCE,
                **{k: north_cl[k].reindex(n.index).to_numpy(float) for k in keys},
            }
        )
    ]
    for name, spec in cfg.extra_sets.items():
        lt, ids, tiles = extra_labels(spec)
        ml = pd.Series(preds[name].mean, index=preds[name].ids).reindex(ids)
        frames.append(
            pd.DataFrame(
                {
                    "z": lt.loc[ids, spec.column].to_numpy(float),
                    "ml": ml.to_numpy(float),
                    "ppub": pub.reindex(ids).to_numpy(float),
                    "zpf": zpf.reindex(ids).to_numpy(float),
                    "poff": phz_values([spec.phz], ids),
                    "tile": tiles,
                    "src": spec.source,
                    **{k: classical[k][name].reindex(ids).to_numpy(float) for k in keys},
                }
            )
        )
    zh = (
        pd.read_parquet(cfg.phosphoros_zpf.hetdex)
        .drop_duplicates("object_id")
        .set_index("object_id")
        .ppub_median
    )
    h_ids, h_tiles = packet_tiles(cfg.hetdex_packet)
    th = pd.Series(h_tiles, index=h_ids)
    frames.append(
        pd.DataFrame(
            {
                "z": hx.z.values,
                "ml": hx.ml.values,
                "ppub": hx.ppub.values,
                "zpf": zh.reindex(hx.index).where(lambda v: v >= 0).to_numpy(float),
                "poff": hx.poff.values,
                "tile": th.reindex(hx.index).to_numpy(),
                "src": HETDEX_SOURCE,
                **{k: classical[k][HETDEX_SET].reindex(hx.index).to_numpy(float) for k in keys},
            }
        )
    )
    methods = ["ml", "ppub", "zpf", "poff", *keys]
    d = pd.concat(frames, ignore_index=True)
    return d[np.isfinite(d[["z", *methods]]).all(axis=1) & (d.z > 0)]


def classical_frames(
    cfg: ReportConfig,
    n: pd.DataFrame,
    hx: pd.DataFrame,
    preds: Mapping[str, SeedStack],
    nt: pd.DataFrame,
    classical: Mapping[str, Mapping[str, pd.Series]],
) -> tuple[pd.DataFrame, list[tuple[str, dict[str, FloatArray], FloatArray, NDArray[Any]]]]:
    keys = list(classical)
    d = n.join(nt[["tile"]], how="inner")
    for k in keys:
        d[k] = pd.concat([classical[k][s] for s in NORTH_SETS]).reindex(d.index).to_numpy(float)
    d = d[np.isfinite(d[keys]).all(axis=1)]
    sets: list[tuple[str, dict[str, FloatArray], FloatArray, NDArray[Any]]] = []
    for name, spec in cfg.extra_sets.items():
        lt, ids, tiles = extra_labels(spec)
        frame = {
            "ml": pd.Series(preds[name].mean, index=preds[name].ids).reindex(ids).to_numpy(float),
            **{k: classical[k][name].reindex(ids).to_numpy(float) for k in keys},
        }
        sets.append((name.upper(), frame, lt.loc[ids, spec.column].to_numpy(float), tiles))
    with h5py.File(cfg.packet, "r") as h:
        vi = h["valid_indices"][:]
        vids, vt = h["object_id"][:][vi], h["owner_tile"][:][vi]
        zv = h["valid_z"][:].astype(float)
    valid = preds[VALID_SET]
    frame = {
        "ml": pd.Series(valid.mean, index=valid.ids).reindex(vids).to_numpy(float),
        **{k: classical[k][VALID_SET].reindex(vids).to_numpy(float) for k in keys},
    }
    sets.append(("F_VALID", frame, zv, vt))
    h_ids, h_tiles = packet_tiles(cfg.hetdex_packet)
    tiles = pd.Series(h_tiles, index=h_ids).reindex(hx.index).to_numpy()
    frame = {
        "ml": hx.ml.values,
        "ppub": hx.ppub.values,
        "poff": hx.poff.values,
        **{k: classical[k][HETDEX_SET].reindex(hx.index).to_numpy(float) for k in keys},
    }
    ok = np.all([np.isfinite(v) for v in frame.values()], 0)
    sets.append(
        ("HETDEX", {k: v[ok] for k, v in frame.items()}, hx.z.values[ok], tiles[ok])
    )
    return d, sets


def zval_inputs(
    cfg: ReportConfig, metas: Mapping[str, Mapping[str, Any]], zval: SeedStack
) -> tuple[FloatArray, dict[str, FloatArray]]:
    first = metas[next(iter(metas))]["zval"]
    ids, z = np.array(first["ids"], np.int64), np.array(first["z"], float)
    check(np.array_equal(zval.ids, ids), "zval predictions follow the meta ids")
    poff = phz_values([cfg.phz_lookup, cfg.cdfs_phz, cfg.ho_phz], ids)
    methods = {
        "ml": zval.mean,
        "ppub": phosphoros_series(cfg.phosphoros.files).reindex(ids).to_numpy(float),
        "zpf": phosphoros_series(cfg.phosphoros_zpf.files).reindex(ids).to_numpy(float),
        "poff": poff,
    }
    return z, methods


def magnitude(flux: FloatArray) -> FloatArray:
    out: FloatArray = np.where(
        flux > 0, MAG_ZERO - 2.5 * np.log10(np.clip(flux, MIN_FLUX, None)), np.nan
    )
    return out


def ho_reference(cfg: ReportConfig, ids: IntArray) -> pd.DataFrame:
    return (
        pd.read_parquet(cfg.ho_reference)
        .drop_duplicates("object_id")
        .set_index("object_id")
        .reindex(ids)
    )


def field_labels(cfg: ReportConfig, hx: pd.DataFrame) -> dict[str, Sample]:
    with h5py.File(cfg.packet, "r") as h:
        idx = np.r_[h["train_indices"][:], h["valid_indices"][:]]
        fz = np.r_[h["train_z"][:].astype(float), h["valid_z"][:].astype(float)]
        vis = json.loads(h.attrs["bands_json"]).index("VIS")
        fflux = h["raw_photometry_flux_ujy"][:][idx, vis]
        known = h["object_id"][:][idx]
        fvt = np.unique(h["owner_tile"][:][h["valid_indices"][:]])
    cl = pd.read_parquet(cfg.cdfs_labels).drop_duplicates("object_id").set_index("object_id")
    cid, ctl = packet_tiles(cfg.cdfs_packet)
    ck = np.isin(cid, cl.index.values) & ~np.isin(ctl, fvt) & ~np.isin(cid, known)
    cz = cl.loc[cid[ck]].z.to_numpy(float)
    cz = cz[np.isfinite(cz) & (cz >= 0)]
    with np.load(cfg.ho_photometry, allow_pickle=False) as nat:
        nat_ids, bands = nat["object_id"], [str(b) for b in nat["bands"]]
        valid, flux, fc, tiles = nat["valid"], nat["flux"], nat["field_code"], nat["tile"]
    ref = ho_reference(cfg, nat_ids)
    hz = ref.z_reference.to_numpy(float)
    ok = (
        (ref.reference_status.to_numpy() == VALID_REFERENCE)
        & np.isfinite(hz)
        & ~ref.reference_outside_model_grid.to_numpy(bool)
    )
    iv = bands.index("VIS")
    hflux = np.where(valid[:, iv], flux[:, iv], np.nan)
    ok &= ~((fc == FIELD_F) & (np.isin(tiles, fvt) | np.isin(nat_ids, np.r_[known, cid[ck]])))
    dz = [
        pd.read_parquet(spec.labels)[spec.column].to_numpy(float)
        for spec in cfg.extra_sets.values()
    ]
    nt_all = pd.read_parquet(cfg.north_labels).drop_duplicates("object_id")
    f_ok, s_ok = ok & (fc == FIELD_F), ok & (fc == FIELD_S)
    return {
        "F": (np.r_[fz, cz, hz[f_ok]], magnitude(np.r_[fflux, hflux[f_ok]])),
        "S": (hz[s_ok], magnitude(hflux[s_ok])),
        "N": (
            np.r_[nt_all.z_strong.to_numpy(float), *dz, hx.z.values],
            np.r_[nt_all.vis.to_numpy(float), hx.vis.values],
        ),
    }


def label_facts(
    cfg: ReportConfig, meta: Mapping[str, Any], n_common: int, n_hetdex: int
) -> dict[str, Any]:
    facts: dict[str, Any] = {}
    with h5py.File(cfg.packet, "r") as h:
        lookup = [
            x.decode() if isinstance(x, bytes) else str(x) for x in h["label_source_lookup"][:]
        ]
        for split in ("train", "valid"):
            c = collections.Counter(h[f"{split}_label_source_code"][:].tolist())
            facts[f"F_{split}_by_source"] = {lookup[k]: int(v) for k, v in sorted(c.items())}
        facts["F_train_n"], facts["F_valid_n"] = int(len(h["train_z"])), int(len(h["valid_z"]))
        facts["bands14"] = json.loads(h.attrs["bands_json"])
    facts["model_deep_counts"], facts["model_n_train"] = meta["deep_counts"], meta["n_train"]
    facts["model_args"] = {k: meta["args"][k] for k in MODEL_ARGS}
    cd = pd.read_parquet(cfg.cdfs_labels)
    facts["CDFS_kind"] = {
        "spec(1)": int((cd.kind == 1).sum()),
        "grism(2)": int((cd.kind == 2).sum()),
    }
    nt = pd.read_parquet(cfg.north_labels).drop_duplicates("object_id")
    by_program = nt.groupby(["source", "program"], dropna=False).size()
    facts["N_labels_by_source_program"] = {f"{s}/{p}": int(v) for (s, p), v in by_program.items()}
    facts["N_subsets"] = nt.subset.value_counts().to_dict()
    facts["N_common_set_n"] = n_common
    for name, spec in cfg.extra_sets.items():
        facts[f"{name}_n"] = int(len(pd.read_parquet(spec.labels)))
    ref = pd.read_parquet(cfg.ho_reference).drop_duplicates("object_id")
    with np.load(cfg.ho_photometry, allow_pickle=False) as nat:
        fc = dict(zip(nat["object_id"].tolist(), nat["field_code"].tolist(), strict=True))
    ref["field"] = [fc.get(int(o), -1) for o in ref.object_id]
    ok = (ref.reference_status == VALID_REFERENCE) & ~ref.reference_outside_model_grid.astype(bool)
    by_family = ref[ok].groupby(["field", "reference_source_family"]).size()
    facts["HO_by_field_family"] = {f"{f}/{s}": int(v) for (f, s), v in by_family.items()}
    facts["HETDEX_primary_n"] = n_hetdex
    with np.load(cfg.store_rows) as rows:
        facts["store_rows_files"] = list(rows.files)
        facts["store"] = {
            "rows": int(len(rows["object_id"])),
            "owners": int(len(np.unique(rows["tile"]))),
            "image_available": int(rows["image_available"].sum()),
            "image_complete": int(rows["image_complete"].sum()),
            "table_available": int(rows["table_available"].sum()),
            "ssl_keep": int(rows["ssl_keep"].sum()),
        }
    return facts


def eval_blocks(
    cfg: ReportConfig, preds: Mapping[str, SeedStack], nt: pd.DataFrame
) -> list[EvalBlock]:
    with h5py.File(cfg.packet, "r") as h:
        vi = h["valid_indices"][:]
        packet_ids, owner = h["object_id"][:], h["owner_tile"][:]
        zv = h["valid_z"][:].astype(float)
    valid = preds[VALID_SET]
    check(np.array_equal(valid.ids, packet_ids[vi]), "valid ids")
    blocks = [
        EvalBlock(
            name="F_VALID",
            sets=(VALID_SET,),
            rows=np.arange(len(valid.ids), dtype=np.int64),
            z=zv,
            tiles=owner[vi],
            phz=phz_values([cfg.phz_lookup], valid.ids),
        )
    ]
    ids_n = np.concatenate([preds[s].ids for s in NORTH_SETS])
    keep = np.flatnonzero(np.isin(ids_n, nt.index.values))
    t = nt.loc[ids_n[keep]]
    zz, tiles, phz = t.z_strong.to_numpy(float), t.tile.to_numpy(), t.phz.to_numpy(float)

    def north_block(name: str, m: NDArray[np.bool_]) -> EvalBlock:
        return EvalBlock(name, NORTH_SETS, keep[m], zz[m], tiles[m], phz[m])

    blocks.append(north_block("N_all", np.ones(len(keep), bool)))
    for sub in N_SUBSETS:
        m = t.subset.to_numpy() == sub
        if m.sum() > MIN_SUBSET:
            blocks.append(north_block(sub, m))
    vis = t.vis.to_numpy(float)
    for lo, hi in N_VIS_BINS:
        blocks.append(north_block(f"N_vis_{lo}_{hi}", (vis >= lo) & (vis < hi)))
    for lo, hi in N_Z_BINS:
        blocks.append(north_block(f"N_z_{lo}_{hi}", (zz >= lo) & (zz < hi)))
    ho = preds[HO_SET]
    h_ids, h_tiles = packet_tiles(cfg.ho_packet)
    check(np.array_equal(h_ids, ho.ids), "ho ids")
    with np.load(cfg.ho_photometry, allow_pickle=False) as nat:
        pos = {int(o): i for i, o in enumerate(nat["object_id"])}
        field = nat["field_code"][np.array([pos[int(o)] for o in ho.ids])]
    ref = pd.read_parquet(cfg.ho_reference).drop_duplicates("object_id").set_index("object_id")
    ref = ref.loc[ho.ids]
    z = ref.z_reference.to_numpy(float)
    ok = (
        (ref.reference_status.to_numpy() == VALID_REFERENCE)
        & np.isfinite(z)
        & ~ref.reference_outside_model_grid.to_numpy(bool)
    )
    ho_phz = phz_values([cfg.ho_phz], ho.ids)
    for name, fc in HO_BLOCKS:
        m = ok & (field == fc)
        blocks.append(EvalBlock(name, (HO_SET,), np.flatnonzero(m), z[m], h_tiles[m], ho_phz[m]))
    for name, spec in cfg.extra_sets.items():
        d = preds[name]
        lt = pd.read_parquet(spec.labels).drop_duplicates("object_id").set_index("object_id")
        p_ids, p_tiles = packet_tiles(spec.packet)
        check(np.array_equal(p_ids, d.ids), f"{name} ids")
        m = np.isin(d.ids, lt.index.values)
        blocks.append(
            EvalBlock(
                name=name.upper(),
                sets=(name,),
                rows=np.flatnonzero(m),
                z=lt.loc[d.ids[m], spec.column].to_numpy(float),
                tiles=p_tiles[m],
                phz=phz_values([spec.phz], d.ids[m]),
            )
        )
    return blocks


def hetdex_inputs(
    cfg: ReportConfig, h6: SeedStack
) -> tuple[FloatArray, NDArray[Any], NDArray[Any], dict[str, FloatArray], FloatArray]:
    ids, tiles = packet_tiles(cfg.hetdex_packet)
    check(np.array_equal(h6.ids, ids), "prediction rows = packet rows")
    stratum = pd.read_parquet(cfg.hetdex_meta).set_index("object_id").loc[ids].stratum.to_numpy()
    z = pd.read_parquet(cfg.hetdex_labels).set_index("object_id").loc[ids].z.to_numpy(float)
    ppub, t = phosphoros_column(cfg.phosphoros.hetdex, ids)
    zpf, _ = phosphoros_column(cfg.phosphoros_zpf.hetdex, ids)
    sp = phosphoros_width(t.ppub_lo68.to_numpy(float), t.ppub_hi68.to_numpy(float))
    sp = np.where(np.isfinite(sp), sp, np.nanmedian(sp))
    poff = phz_values([cfg.hetdex_phz], ids)
    comps = {"P_pub": ppub, "P_zpf": zpf, "P_off": np.where(np.isfinite(poff), poff, np.nan)}
    return z, tiles, stratum, comps, sp


def fill_width(sp: FloatArray) -> FloatArray:
    fill = np.nanmedian(sp) if np.isfinite(sp).any() else WIDTH_DEFAULT
    out: FloatArray = np.where(np.isfinite(sp), sp, fill)
    return out


def block_values(
    block: EvalBlock, variant: Mapping[str, SetPair]
) -> tuple[FloatArray, FloatArray]:
    z = np.concatenate([variant[s][0] for s in block.sets], 1)[:, block.rows]
    ppub = np.concatenate([variant[s][1] for s in block.sets])[block.rows]
    return z, ppub
