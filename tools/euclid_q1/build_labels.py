from __future__ import annotations

import glob
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import click
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.io import fits
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.spatial import cKDTree

from lib.dataloaders.euclid_transforms import BANDS, validate_band_order
from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as

logger = get_logger()

IR_CATALOG = "goodss_3dhst.v4.1.5.IR.cat"
ZFIT_CATALOG = "goodss_3dhst.v4.1.5.zfit.linematched.fits"
CDFS_RA = (52.9, 53.4)
CDFS_DEC = (-28.05, -27.55)
MATCH_ARCSEC = 0.5
NEXT_ARCSEC = 1.0
SPEC_Z_MIN = 0.01
GRISM_WIDTH_MAX = 0.02
TILE_PATTERN = re.compile(r"TILE(\d+)")
NORTH_LABEL_COLUMNS = ["object_id", "z_strong", "canonical_source_key", "tile"]
NORTH_LABEL_FILTERS = [
    ("field_code", "=", 0),
    ("label_accepted", "=", True),
    ("label_tier", "=", "EXTERNAL_STRONG"),
]
NORTH_MAGS = {"r": "R_EXT_MEGACAM", "z": "Z_EXT_HSC", "u": "U_EXT_MEGACAM", "vis": "VIS"}
REFERENCE_COLUMNS = [
    "object_id",
    "z_reference",
    "reference_status",
    "reference_outside_model_grid",
]
TRANSLATION_SETS = ("F_TRAIN", "F_VALID", "CDFS", "MUSE", "HO_F_SDSS")
FAMILY = {"MUSE": "MUSE", "HO_F_SDSS": "SDSS"}

type Arrays = dict[str, NDArray[Any]]


class CdfsLabelsConfig(BaseModel):
    threedhst_dir: Path
    mer_dir: Path
    edf_f_tiles: Path
    packet: Path
    ho_native: Path
    output_dir: Path
    workers: int = 8


class NorthLabelsConfig(BaseModel):
    packet: Path
    nb_packet: Path
    labels: Path
    desi: Path
    native: Path
    phz_lookup: Path
    output: Path


class TranslationLabelsConfig(BaseModel):
    packet: Path
    cdfs_packet: Path
    ho_packet: Path
    cdfs_labels: Path
    ho_native: Path
    reference: Path
    phz_lookup: Path
    output_dir: Path


def xyz(ra: NDArray[np.floating], dec: NDArray[np.floating]) -> NDArray[np.float64]:
    r, d = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], 1)


def tile_of(path: str) -> str:
    match = TILE_PATTERN.search(path)
    if match is None:
        raise ValueError(f"no tile id in {path}")
    return match.group(1)


def threedhst_redshifts(directory: Path) -> Arrays:
    catalog = directory / IR_CATALOG
    cols: dict[str, int] = {}
    with catalog.open() as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            p = line.split()
            cols[p[2]] = int(p[1]) - 1
    ir = np.loadtxt(catalog, usecols=(cols["NUMBER"], cols["X_WORLD"], cols["Y_WORLD"]))
    pos = {int(n): (ra, de) for n, ra, de in ir}
    with fits.open(directory / ZFIT_CATALOG) as hdus:
        z = hdus[1].data
        zs = np.asarray(z["z_spec"], float)
        zb = np.asarray(z["z_best"], float)
        zbs = np.asarray(z["z_best_s"], int)
        ug = np.asarray(z["use_zgrism"], int)
        w68 = (np.asarray(z["z_best_u68"], float) - np.asarray(z["z_best_l68"], float)) / (1 + zb)
        phot_id = np.asarray(z["phot_id"], int)
        jh = np.asarray(z["jh_mag"], float)
    spec = zs > SPEC_Z_MIN
    grism = (zbs == 2) & (ug == 1) & (w68 < GRISM_WIDTH_MAX) & (zb > SPEC_Z_MIN)
    keep = spec | grism
    ids = phot_id[keep]
    logger.info(
        "3D-HST rows %d spec %d grism %d kept %d",
        len(zs),
        int(spec.sum()),
        int((grism & ~spec).sum()),
        int(keep.sum()),
    )
    return dict(
        phot_id=ids,
        ra=np.array([pos[i][0] for i in ids]),
        dec=np.array([pos[i][1] for i in ids]),
        z=np.where(spec[keep], zs[keep], zb[keep]),
        kind=np.where(spec[keep], 1, 2),
        jh=jh[keep],
    )


def read_cdfs_tile(path: str) -> Arrays | None:
    with fits.open(path, memmap=True) as hdus:
        d = hdus[1].data
        ra = np.asarray(d["RIGHT_ASCENSION"], float)
        de = np.asarray(d["DECLINATION"], float)
        m = (ra > CDFS_RA[0]) & (ra < CDFS_RA[1]) & (de > CDFS_DEC[0]) & (de < CDFS_DEC[1])
        if not m.any():
            return None
        return dict(
            object_id=np.asarray(d["OBJECT_ID"]).astype(np.int64)[m],
            ra=ra[m],
            dec=de[m],
            flux=np.stack([np.asarray(d[f"FLUX_{b}_2FWHM_APER"], float)[m] for b in BANDS], 1),
            error=np.stack([np.asarray(d[f"FLUXERR_{b}_2FWHM_APER"], float)[m] for b in BANDS], 1),
            tile=np.full(int(m.sum()), int(tile_of(path)), np.int64),
        )


def cdfs_mer(config: CdfsLabelsConfig) -> Arrays:
    tiles = set(config.edf_f_tiles.read_text().split())
    files = sorted(f for f in glob.glob(str(config.mer_dir / "*.fits")) if tile_of(f) in tiles)
    with ProcessPoolExecutor(config.workers) as pool:
        parts = [p for p in pool.map(read_cdfs_tile, files) if p is not None]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    _, first = np.unique(merged["object_id"], return_index=True)
    return {k: v[first] for k, v in merged.items()}


def build_cdfs(config: CdfsLabelsConfig) -> tuple[Arrays, Arrays, pd.DataFrame]:
    t = threedhst_redshifts(config.threedhst_dir)
    mer = cdfs_mer(config)
    tree = cKDTree(xyz(mer["ra"], mer["dec"]))
    dist, idx = tree.query(xyz(t["ra"], t["dec"]), k=2)
    sep = np.degrees(2 * np.arcsin(np.clip(dist / 2, 0, 1))) * 3600
    good = (sep[:, 0] <= MATCH_ARCSEC) & (sep[:, 1] >= NEXT_ARCSEC)
    df = pd.DataFrame(
        dict(
            row=idx[good, 0],
            z=t["z"][good],
            kind=t["kind"][good],
            jh=t["jh"][good],
            sep=sep[good, 0],
        )
    )
    df = df.drop_duplicates("row", keep=False)
    with h5py.File(config.packet, "r") as h:
        packet_ids = h["object_id"][:]
    with np.load(config.ho_native, allow_pickle=False) as z:
        ho_ids = z["object_id"]
    oid = mer["object_id"][df.row.to_numpy()]
    keep = ~np.isin(oid, np.union1d(packet_ids, ho_ids))
    df, oid = df[keep], oid[keep]
    r = df.row.to_numpy()
    valid = np.isfinite(mer["flux"][r]) & np.isfinite(mer["error"][r]) & (mer["error"][r] > 0)
    ok4 = valid[:, :4].all(1)
    df, oid, r, valid = df[ok4], oid[ok4], r[ok4], valid[ok4]
    o = np.argsort(oid)
    df, oid, r, valid = df.iloc[o], oid[o], r[o], valid[o]
    native = dict(
        object_id=oid,
        field_code=np.ones(len(oid), np.uint8),
        tile=mer["tile"][r],
        ra_deg=mer["ra"][r],
        dec_deg=mer["dec"][r],
        flux=mer["flux"][r],
        error=mer["error"][r],
        valid=valid,
        bands=np.array(BANDS),
    )
    labels = pd.DataFrame(
        dict(
            object_id=oid,
            tile=mer["tile"][r],
            z=df.z.to_numpy(),
            kind=df.kind.to_numpy(),
            jh=df.jh.to_numpy(),
            sep=df.sep.to_numpy(),
        )
    )
    logger.info(
        "CDFS input %d matched %d outside packets %d with euclid4 %d kinds %s tiles %d",
        len(t["z"]),
        int(good.sum()),
        int(keep.sum()),
        len(oid),
        dict(zip(*np.unique(labels.kind, return_counts=True), strict=True)),
        len(np.unique(labels.tile)),
    )
    return t, native, labels


def mags(
    flux: NDArray[np.floating], valid: NDArray[np.bool_], bands: list[str]
) -> dict[str, NDArray[np.float64]]:
    out = {}
    for name, band in NORTH_MAGS.items():
        k = bands.index(band)
        x = flux[:, k]
        ok = valid[:, k] & (x > 0)
        m = np.full(len(x), np.nan)
        m[ok] = 23.9 - 2.5 * np.log10(x[ok])
        out[name] = m
    return out


def build_north(config: NorthLabelsConfig) -> pd.DataFrame:
    with h5py.File(config.packet, "r") as h:
        ids_a = h["object_id"][:][h["test_indices"][:]]
    with h5py.File(config.nb_packet, "r") as h:
        ids_b = h["object_id"][:]
    df = pd.DataFrame(
        dict(object_id=np.r_[ids_a, ids_b], subset=["N-A"] * len(ids_a) + ["N-B"] * len(ids_b))
    )
    lab = pq.read_table(
        config.labels, columns=NORTH_LABEL_COLUMNS, filters=NORTH_LABEL_FILTERS
    ).to_pandas()
    df = df.merge(lab, on="object_id", how="left", validate="one_to_one")
    check(df.z_strong.notna().all(), "every N packet object has a strong label")
    df["source"] = df.canonical_source_key.str.split(":").str[0]
    df["tid"] = df.canonical_source_key.str.split(":").str[1]
    desi = pq.read_table(config.desi, columns=["targetid", "spectype", "program"]).to_pandas()
    desi["tid"] = desi.targetid.astype(str)
    df = df.merge(desi[["tid", "spectype", "program"]].drop_duplicates("tid"), on="tid", how="left")
    with np.load(config.native, allow_pickle=False) as z:
        nat = {k: z[k] for k in ("object_id", "flux", "valid", "bands")}
    pos = {int(o): i for i, o in enumerate(nat["object_id"])}
    j = np.array([pos[int(o)] for o in df.object_id])
    m = mags(nat["flux"][j], nat["valid"][j], list(nat["bands"]))
    df["r"], df["rz"], df["ur"], df["vis"] = m["r"], m["r"] - m["z"], m["u"] - m["r"], m["vis"]
    with np.load(config.phz_lookup, allow_pickle=False) as z:
        ph = {k: z[k] for k in ("object_id", "PHZ_MEDIAN", "PHZ_70_INT", "PHZ_CLASSIFICATION")}
    pmap = {int(o): i for i, o in enumerate(ph["object_id"])}
    k = np.array([pmap.get(int(o), -1) for o in df.object_id])
    has = k >= 0
    df["phz"] = np.where(has, ph["PHZ_MEDIAN"].reshape(-1)[np.maximum(k, 0)], np.nan)
    i70 = ph["PHZ_70_INT"].reshape(len(ph["object_id"]), -1)
    df["phz_w70"] = np.where(has, (i70[:, -1] - i70[:, 0])[np.maximum(k, 0)], np.nan)
    df["phz_class"] = np.where(has, ph["PHZ_CLASSIFICATION"].reshape(-1)[np.maximum(k, 0)], -1)
    logger.info("N labels %s", df.subset.value_counts().to_dict())
    return df


def ho_sets(config: TranslationLabelsConfig) -> tuple[dict[str, Arrays], Arrays]:
    with h5py.File(config.ho_packet, "r") as h:
        validate_band_order(json.loads(h.attrs["bands_json"]))
        h_oid = h["object_id"][:]
        h_field = h["field_code"][:]
    keys = ("object_id", "bands", "flux", "error", "valid", "source_family", "tile")
    with np.load(config.ho_native, allow_pickle=False) as z:
        hn = {k: z[k] for k in keys}
    pos = {int(o): i for i, o in enumerate(hn["object_id"])}
    j = np.array([pos[int(o)] for o in h_oid])
    fam = hn["source_family"][j].astype(str)
    raw = dict(
        object_id=h_oid,
        ho_row=np.arange(len(h_oid), dtype=np.int64),
        field_code=h_field,
        source_family=fam,
        tile=hn["tile"][j],
        bands=hn["bands"],
        flux=hn["flux"][j],
        error=hn["error"][j],
        valid=hn["valid"][j],
    )
    muse = np.where((h_field == 1) & (fam == "MUSE"))[0]
    fsdss = np.where((h_field == 1) & (fam != "MUSE"))[0]
    ids_f = h_oid[np.concatenate([muse, fsdss])]
    ref = pq.read_table(
        config.reference, columns=REFERENCE_COLUMNS, filters=[("object_id", "in", ids_f.tolist())]
    ).to_pandas()
    logger.info(
        "HO references %d rows, %d duplicated", len(ref), int(ref.object_id.duplicated().sum())
    )
    ref = ref.drop_duplicates("object_id").set_index("object_id")
    sets = {}
    for name, rows in (("MUSE", muse), ("HO_F_SDSS", fsdss)):
        r = ref.reindex(h_oid[rows])
        sets[name] = dict(
            rows=rows.astype(np.int64),
            object_id=h_oid[rows],
            z=r.z_reference.to_numpy(np.float64),
            status=r.reference_status.fillna("missing").to_numpy(str),
            outside=r.reference_outside_model_grid.fillna(True).to_numpy(bool),
        )
    return sets, raw


def build_translation(config: TranslationLabelsConfig) -> tuple[Arrays, Arrays, Arrays]:
    sets: dict[str, Arrays] = {}
    with h5py.File(config.packet, "r") as h:
        validate_band_order(json.loads(h.attrs["bands_json"]))
        oid = h["object_id"][:]
        lookup = [
            x.decode() if isinstance(x, bytes) else str(x) for x in h["label_source_lookup"][:]
        ]
        for name, split in (("F_TRAIN", "train"), ("F_VALID", "valid")):
            rows = h[f"{split}_indices"][:].astype(np.int64)
            sets[name] = dict(
                rows=rows,
                object_id=oid[rows],
                z=h[f"{split}_z"][:].astype(np.float64),
                tier=h[f"{split}_label_tier"][:].astype(np.int64),
                code=h[f"{split}_label_source_code"][:].astype(np.int64),
            )
    with h5py.File(config.cdfs_packet, "r") as h:
        c_oid = h["object_id"][:]
    lab = pq.read_table(config.cdfs_labels, columns=["object_id", "z", "kind"]).to_pandas()
    check(
        not lab.object_id.duplicated().any() and set(c_oid.tolist()) <= set(lab.object_id.tolist()),
        "cdfs labels coverage",
    )
    lab = lab.set_index("object_id").loc[c_oid]
    sets["CDFS"] = dict(
        rows=np.arange(len(c_oid), dtype=np.int64),
        object_id=c_oid,
        z=lab.z.to_numpy(np.float64),
        kind=lab.kind.to_numpy(np.int64),
    )
    ho, ho_raw = ho_sets(config)
    sets |= ho
    packet, cdfs_h5, ho_h5 = config.packet.name, config.cdfs_packet.name, config.ho_packet.name
    z_source = {
        "F_TRAIN": f"{packet}:train_z",
        "F_VALID": f"{packet}:valid_z",
        "CDFS": f"{config.cdfs_labels.name}:z",
    }
    source_h5 = {"F_TRAIN": packet, "F_VALID": packet, "CDFS": cdfs_h5}
    flat: Arrays = {}
    for name in TRANSLATION_SETS:
        d = sets[name]
        n = len(d["object_id"])
        out = dict(
            object_id=d["object_id"].astype(np.int64),
            source_row=d["rows"].astype(np.int64),
            z_fix=d["z"],
            z_source=np.full(n, z_source.get(name, f"{config.reference.name}:z_reference")),
            source_h5=np.full(n, source_h5.get(name, ho_h5)),
            label_tier=d.get("tier", np.full(n, -1)).astype(np.int64),
            label_source_code=d.get("code", np.full(n, -1)).astype(np.int64),
            kind=d.get("kind", np.full(n, -1)).astype(np.int64),
            reference_status=d.get("status", np.full(n, "n/a")),
            reference_outside_grid=d.get("outside", np.zeros(n, bool)),
            family=np.full(n, FAMILY.get(name, name)),
        )
        flat |= {f"{name}__{k}": v for k, v in out.items()}
        logger.info("%s rows %d finite z %d", name, n, int(np.isfinite(d["z"]).sum()))
    flat["label_source_lookup"] = np.array(lookup)
    with np.load(config.phz_lookup, allow_pickle=False) as z:
        phz = dict(object_id=z["object_id"], PHZ_MEDIAN=z["PHZ_MEDIAN"])
    return flat, ho_raw, phz


CONFIG = click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)


@click.group()
def cli() -> None:
    pass


@cli.command()
@CONFIG
def cdfs(config_path: Path) -> None:
    config = load_yaml_as(str(config_path), CdfsLabelsConfig)
    t, native, labels = build_cdfs(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(config.output_dir / "threedhst_z.npz", **dict[str, Any](t))
    np.savez(config.output_dir / "cdfs_native_photometry.npz", **dict[str, Any](native))
    labels.to_parquet(config.output_dir / "cdfs_train.parquet")
    logger.info("wrote CDFS labels to %s", config.output_dir)


@cli.command()
@CONFIG
def north(config_path: Path) -> None:
    config = load_yaml_as(str(config_path), NorthLabelsConfig)
    df = build_north(config)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.output)
    logger.info("wrote %s (%d rows)", config.output, len(df))


@cli.command()
@CONFIG
def translation(config_path: Path) -> None:
    config = load_yaml_as(str(config_path), TranslationLabelsConfig)
    flat, ho_raw, phz = build_translation(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(config.output_dir / "f_labels.npz", **dict[str, Any](flat))
    np.savez(config.output_dir / "ho_raw.npz", **dict[str, Any](ho_raw))
    np.savez(config.output_dir / "phz_subset.npz", **dict[str, Any](phz))
    logger.info("wrote translation labels to %s", config.output_dir)


if __name__ == "__main__":
    cli()
