from __future__ import annotations

import glob
import json
import re
import shutil
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, cast

import click
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.spatial import cKDTree

from lib.dataloaders.euclid_transforms import (
    BANDS,
    IMAGE_BANDS,
    IMAGE_SIDE,
    SPECTRUM_LENGTH,
    Normalizers,
    image_usable,
    load_normalizers,
    transform_image,
    transform_table,
    validate_band_order,
)
from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as

logger = get_logger()

FIELD = "nep"
STRATA = ("oii", "lae", "agn", "lzg")
MIN_PROBABILITY = 0.5
MATCH_ARCSEC = 1.0
NEXT_ARCSEC = 1.5
EXCLUSION_ARCSEC = 1.0
CATALOG_COLUMNS = {
    "id": "source_id",
    "ra": "RA",
    "dec": "DEC",
    "field": "field",
    "source_type": "source_type",
    "p_conf": "p_conf",
    "p_cnn": "p_cnn",
    "line_sn": "sn",
    "line_flux": "flux_oii",
}
LABEL_COLUMNS = {"z": "z_hetdex", "z_conf": "z_hetdex_conf"}
IDS_FILE = "h6_ids.npy"
META_FILE = "h6_meta.parquet"
LABELS_FILE = "h6_labels.parquet"
MOSAIC_BANDS = {
    "VIS": ("VIS", "VIS"),
    "Y": ("NISP", "NIR-Y"),
    "J": ("NISP", "NIR-J"),
    "H": ("NISP", "NIR-H"),
}
MOSAIC_KINDS = ("science", "rms", "flag")
EXPECTED_FILTER = {"VIS": "VIS", "Y": "NIR_Y", "J": "NIR_J", "H": "NIR_H"}
GEOMETRY_KEYS = (
    "CTYPE1",
    "CTYPE2",
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
    "NAXIS1",
    "NAXIS2",
)
IMAGE_KEYS = ("image_flux_uJy", "image_rms_uJy", "image_valid", "image_flags", "image_band_present")
AB_ZERO_POINT = 23.9
HALF = IMAGE_SIDE // 2
TABLE_SLOTS = len(BANDS)

type Arrays = dict[str, NDArray[Any]]


class IdSource(BaseModel):
    path: Path
    column: str = "object_id"


class PositionSource(BaseModel):
    path: Path
    ra: str = "ra"
    dec: str = "dec"


class HetdexConfig(BaseModel):
    catalog: Path | None = None
    mer_dir: Path | None = None
    edfn_tiles: Path | None = None
    catalog_accuracy: Path | None = None
    exclusion_ids: list[IdSource] = []
    exclusion_positions: list[PositionSource] = []
    build_dir: Path | None = None
    native: Path | None = None
    mosaic_dir: Path | None = None
    stamp_dir: Path | None = None
    normalizers: Path | None = None
    packet: Path | None = None
    exclude_ids: list[int] = []
    phosphoros_request: Path | None = None
    s3_url: str = "https://nasa-irsa-euclid-q1.s3.amazonaws.com"
    s3_prefix: str = "q1/MER"
    workers: int = 8
    download_workers: int = 4
    download_attempts: int = 5


def need[T](value: T | None, key: str) -> T:
    check(value is not None, f"config key '{key}' is required")
    return cast(T, value)


def load_config(path: Path) -> HetdexConfig:
    return load_yaml_as(str(path), HetdexConfig)


def read_ids(source: IdSource) -> NDArray[np.int64]:
    path, column = source.path, source.column
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        values = pq.read_table(path, columns=[column]).column(column).to_numpy()
    elif suffix == ".npy":
        values = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as z:
            values = z[column]
    elif suffix == ".csv":
        values = pd.read_csv(path, usecols=[column], dtype={column: "int64"})[column].to_numpy()
    elif suffix == ".json":
        entry = json.loads(path.read_text())[column]
        groups = list(entry.values()) if isinstance(entry, dict) else [entry]
        values = np.array([o for group in groups for o in group], np.int64)
    else:
        raise ValueError(f"unsupported id source {path}")
    out: NDArray[np.int64] = np.asarray(values, np.int64)
    return out


def read_positions(source: PositionSource) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    path = source.path
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        t = pq.read_table(path, columns=[source.ra, source.dec]).to_pandas()
        ra, dec = t[source.ra].to_numpy(float), t[source.dec].to_numpy(float)
    elif suffix == ".npz":
        with np.load(path) as z:
            ra, dec = z[source.ra].astype(float), z[source.dec].astype(float)
    elif suffix == ".csv":
        t = pd.read_csv(path, usecols=[source.ra, source.dec])
        ra, dec = t[source.ra].to_numpy(float), t[source.dec].to_numpy(float)
    else:
        raise ValueError(f"unsupported position source {path}")
    return np.asarray(ra, np.float64), np.asarray(dec, np.float64)


def collect_exclusions(
    cfg: HetdexConfig,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
    ids = [read_ids(s) for s in cfg.exclusion_ids]
    positions = [read_positions(s) for s in cfg.exclusion_positions]
    for source, values in zip(cfg.exclusion_ids, ids, strict=True):
        logger.info(
            "exclusion ids %s[%s]: %d unique", source.path, source.column, len(np.unique(values))
        )
    oid: NDArray[np.int64] = np.unique(np.concatenate(ids)) if ids else np.array([], np.int64)
    ra = np.concatenate([p[0] for p in positions]) if positions else np.array([], np.float64)
    dec = np.concatenate([p[1] for p in positions]) if positions else np.array([], np.float64)
    ok = np.isfinite(ra) & np.isfinite(dec)
    return oid, ra[ok], dec[ok]


def unit_vectors(ra: NDArray[np.floating], dec: NDArray[np.floating]) -> NDArray[np.float64]:
    r, d = np.radians(ra), np.radians(dec)
    out: NDArray[np.float64] = np.stack(
        [np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)], 1
    )
    return out


def chord_arcsec(chord: NDArray[np.floating]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = np.degrees(2 * np.arcsin(np.clip(chord / 2, 0, 1))) * 3600
    return out


def tile_name(path: str | Path) -> str:
    match = re.search(r"TILE(\d+)", Path(path).name)
    check(match is not None, f"no tile id in {path}")
    return cast(re.Match[str], match).group(1)


def mer_files(mer_dir: Path, tiles: set[str]) -> list[str]:
    return sorted(f for f in glob.glob(str(mer_dir / "*.fits")) if tile_name(f) in tiles)


def read_mer(
    path: str,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        oid: NDArray[np.int64] = np.asarray(d["OBJECT_ID"]).astype(np.int64)
        ra: NDArray[np.float64] = np.asarray(d["RIGHT_ASCENSION"], np.float64)
        dec: NDArray[np.float64] = np.asarray(d["DECLINATION"], np.float64)
        tile = np.full(len(d), int(tile_name(path)), np.int64)
    return oid, ra, dec, tile


def lowered(values: NDArray[Any]) -> NDArray[np.str_]:
    out: NDArray[np.str_] = np.char.lower(np.char.strip(np.asarray(values).astype(str)))
    return out


def select_objects(cfg: HetdexConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = need(cfg.catalog, "catalog")
    with fits.open(catalog, memmap=True) as h:
        d = h[1].data
        c = {k: np.asarray(d[v]) for k, v in CATALOG_COLUMNS.items()}
    logger.info("catalog rows: %d", len(c["id"]))
    field, stype = lowered(c["field"]), lowered(c["source_type"])
    sel = (
        (field == FIELD)
        & np.isin(stype, STRATA)
        & (c["p_conf"].astype(float) >= MIN_PROBABILITY)
        & (c["p_cnn"].astype(float) >= MIN_PROBABILITY)
    )
    logger.info(
        "field %s, type in %s, p_conf>=%.1f, p_cnn>=%.1f: %d %s",
        FIELD,
        "/".join(STRATA),
        MIN_PROBABILITY,
        MIN_PROBABILITY,
        int(sel.sum()),
        {t: int((sel & (stype == t)).sum()) for t in STRATA},
    )
    ra, dec = c["ra"].astype(float).copy(), c["dec"].astype(float).copy()
    rows = np.flatnonzero(sel & np.isfinite(ra) & np.isfinite(dec))
    tiles = set(need(cfg.edfn_tiles, "edfn_tiles").read_text().split())
    files = mer_files(need(cfg.mer_dir, "mer_dir"), tiles)
    with Pool(cfg.workers) as pool:
        parts = pool.map(read_mer, files)
    oid = np.concatenate([p[0] for p in parts])
    mra = np.concatenate([p[1] for p in parts])
    mdec = np.concatenate([p[2] for p in parts])
    mtile = np.concatenate([p[3] for p in parts])
    tree = cKDTree(unit_vectors(mra, mdec))
    dist, idx = tree.query(unit_vectors(ra[rows], dec[rows]), k=2)
    sep = chord_arcsec(dist)
    good = (sep[:, 0] <= MATCH_ARCSEC) & (sep[:, 1] >= NEXT_ARCSEC)
    logger.info(
        'unique MER match (<=%.1f", next >=%.1f"): %d, no match %d, ambiguous %d',
        MATCH_ARCSEC,
        NEXT_ARCSEC,
        int(good.sum()),
        int((sep[:, 0] > MATCH_ARCSEC).sum()),
        int(((sep[:, 0] <= MATCH_ARCSEC) & (sep[:, 1] < NEXT_ARCSEC)).sum()),
    )
    matched = rows[good]
    nearest = idx[good, 0]
    m = pd.DataFrame(
        dict(
            row=matched,
            object_id=oid[nearest],
            tile=mtile[nearest],
            sep=sep[good, 0],
            stratum=stype[matched],
            hetdex_id=c["id"][matched],
            p_conf=c["p_conf"][matched].astype(float),
            p_cnn=c["p_cnn"][matched].astype(float),
            mer_ra=mra[nearest],
            mer_dec=mdec[nearest],
        )
    )
    for k in ("line_sn", "line_flux"):
        m[k] = c[k][m.row.to_numpy()].astype(float)
    m = m.drop_duplicates("object_id", keep=False)
    logger.info("one HETDEX source per MER object: %d", len(m))
    ex_ids, ex_ra, ex_dec = collect_exclusions(cfg)
    by_id = m.object_id.isin(ex_ids).to_numpy()
    used = np.isin(oid, ex_ids)
    radius = np.radians(EXCLUSION_ARCSEC / 3600)
    q = unit_vectors(m.mer_ra.to_numpy(), m.mer_dec.to_numpy())
    near_used = np.zeros(len(m), bool)
    if used.any():
        used_tree = cKDTree(unit_vectors(mra[used], mdec[used]))
        near_used |= np.isfinite(used_tree.query(q, k=1, distance_upper_bound=radius)[0])
    ext_tree = cKDTree(unit_vectors(ex_ra, ex_dec))
    near_ext = np.isfinite(ext_tree.query(q, k=1, distance_upper_bound=radius)[0])
    drop = by_id | near_used | near_ext
    logger.info(
        'exclude used labels by id and within %.1f": %d (by id %d, near used MER %d, '
        "near external %d; %d unique ids, %d positions)",
        EXCLUSION_ARCSEC,
        int((~drop).sum()),
        int(by_id.sum()),
        int((near_used & ~by_id).sum()),
        int((near_ext & ~by_id & ~near_used).sum()),
        len(ex_ids),
        len(ex_ra),
    )
    m = m[~drop].sort_values("object_id").reset_index(drop=True)
    acc = (
        pq.read_table(
            need(cfg.catalog_accuracy, "catalog_accuracy"),
            columns=["object_id", "vis_asinh", "phz_class"],
            filters=[("object_id", "in", m.object_id.tolist())],
        )
        .to_pandas()
        .drop_duplicates("object_id")
        .set_index("object_id")
    )
    m["vis"] = acc.reindex(m.object_id).vis_asinh.to_numpy(float)
    m["phz_class"] = acc.reindex(m.object_id).phz_class.to_numpy()
    ids = m.object_id.to_numpy(np.int64)
    with fits.open(catalog, memmap=True) as h:
        d = h[1].data
        zr = m.row.to_numpy()
        labels = pd.DataFrame(
            dict(
                object_id=ids,
                stratum=m.stratum.to_numpy(),
                z=np.asarray(d[LABEL_COLUMNS["z"]])[zr].astype(float),
            )
        )
        labels["z_conf"] = np.asarray(d[LABEL_COLUMNS["z_conf"]])[zr].astype(float)
    primary = m.stratum.to_numpy() == STRATA[0]
    logger.info(
        "selected %d objects in %d tiles; [OII] %d in %d tiles; %s",
        len(m),
        m.tile.nunique(),
        int(primary.sum()),
        m.tile[primary].nunique(),
        {t: int((m.stratum == t).sum()) for t in STRATA},
    )
    return m.drop(columns=["row"]), labels


def write_selection(cfg: HetdexConfig) -> None:
    out = need(cfg.build_dir, "build_dir")
    out.mkdir(parents=True, exist_ok=False)
    meta, labels = select_objects(cfg)
    np.save(out / IDS_FILE, meta.object_id.to_numpy(np.int64))
    meta.to_parquet(out / META_FILE)
    labels.to_parquet(out / LABELS_FILE)
    logger.info("selection -> %s", out)


def read_native(path: str, ids: NDArray[np.int64]) -> Arrays | None:
    with fits.open(path, memmap=True) as h:
        d = h[1].data
        oid = np.asarray(d["OBJECT_ID"]).astype(np.int64)
        keep = np.isin(oid, ids)
        if not keep.any():
            return None
        return dict(
            object_id=oid[keep],
            ra=np.asarray(d["RIGHT_ASCENSION"], np.float64)[keep],
            dec=np.asarray(d["DECLINATION"], np.float64)[keep],
            flux=np.stack(
                [np.asarray(d[f"FLUX_{b}_2FWHM_APER"], np.float64)[keep] for b in BANDS], 1
            ),
            error=np.stack(
                [np.asarray(d[f"FLUXERR_{b}_2FWHM_APER"], np.float64)[keep] for b in BANDS], 1
            ),
            tile=np.full(int(keep.sum()), int(tile_name(path)), np.int64),
        )


def native_photometry(cfg: HetdexConfig) -> Arrays:
    build = need(cfg.build_dir, "build_dir")
    ids = np.load(build / IDS_FILE)
    tiles = set(pd.read_parquet(build / META_FILE).tile.astype(str))
    files = mer_files(need(cfg.mer_dir, "mer_dir"), tiles)
    with Pool(cfg.workers) as pool:
        parts = [p for p in pool.map(partial(read_native, ids=ids), files) if p is not None]
    r = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.argsort(r["object_id"])
    r = {k: v[order] for k, v in r.items()}
    check(np.array_equal(r["object_id"], ids), "MER photometry for every selected id once")
    valid = np.isfinite(r["flux"]) & np.isfinite(r["error"]) & (r["error"] > 0)
    logger.info("%d objects; band presence %s", len(ids), valid.mean(0).round(3).tolist())
    return dict(
        object_id=r["object_id"],
        field_code=np.zeros(len(ids), np.uint8),
        tile=r["tile"],
        ra_deg=r["ra"],
        dec_deg=r["dec"],
        flux=r["flux"],
        error=r["error"],
        valid=valid,
        bands=np.array(BANDS),
    )


def write_native(cfg: HetdexConfig) -> None:
    out = need(cfg.native, "native")
    arrays = native_photometry(cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as f:
        np.savez(f, **cast(dict[str, Any], arrays))
    logger.info("native photometry -> %s", out)


def native_tiles(cfg: HetdexConfig, selected: tuple[int, ...] = ()) -> list[int]:
    with np.load(need(cfg.native, "native")) as z:
        tiles = sorted(int(t) for t in np.unique(z["tile"]))
    check(set(selected) <= set(tiles), "requested tiles are not owner tiles of the selection")
    return [t for t in tiles if not selected or t in selected]


def list_keys(base: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        query = {"list-type": "2", "prefix": prefix}
        if token:
            query["continuation-token"] = token
        url = f"{base.rstrip('/')}/?{urllib.parse.urlencode(query)}"
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        ns = root.tag[: root.tag.index("}") + 1] if root.tag.startswith("{") else ""
        keys += [e.text for e in root.iter(ns + "Key") if e.text]
        if root.findtext(ns + "IsTruncated") != "true":
            return keys
        token = root.findtext(ns + "NextContinuationToken")


def mosaic_keys(cfg: HetdexConfig, tile: int) -> dict[tuple[str, str], str]:
    names: dict[str, list[str]] = {}
    for folder in sorted({f for f, _ in MOSAIC_BANDS.values()}):
        prefix = f"{cfg.s3_prefix.strip('/')}/{tile}/{folder}/"
        names[folder] = [k for k in list_keys(cfg.s3_url, prefix) if k.endswith(".fits")]
    keys: dict[tuple[str, str], str] = {}
    for band, (folder, tag) in MOSAIC_BANDS.items():
        starts = {
            "science": f"EUC_MER_BGSUB-MOSAIC-{tag}_TILE{tile}-",
            "rms": f"EUC_MER_MOSAIC-{tag}-RMS_TILE{tile}-",
            "flag": f"EUC_MER_MOSAIC-{tag}-FLAG_TILE{tile}-",
        }
        for kind in MOSAIC_KINDS:
            found = sorted(
                k for k in names[folder] if k.rsplit("/", 1)[-1].startswith(starts[kind])
            )
            check(len(found) == 1, f"tile {tile}: expected one {band}/{kind} mosaic, got {found}")
            keys[(band, kind)] = found[0]
    return keys


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def download(url: str, dst: Path, attempts: int) -> None:
    if dst.exists():
        return
    size = remote_size(url)
    part = dst.with_suffix(".part")
    dst.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(attempts):
        have = part.stat().st_size if part.exists() else 0
        if have == size:
            break
        headers = {"Range": f"bytes={have}-"} if 0 < have < size else {}
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=600) as response:
                append = bool(headers) and response.status == 206
                with part.open("ab" if append else "wb") as f:
                    shutil.copyfileobj(response, f, 1 << 24)
        except OSError as error:
            logger.warning("%s: %s", dst, error)
            time.sleep(10)
    check(part.exists() and part.stat().st_size == size, f"size mismatch {dst}")
    part.rename(dst)
    logger.info("%s (%d bytes)", dst, size)


def mosaic_path(mosaic_dir: Path, tile: int, band: str, kind: str) -> Path:
    return mosaic_dir / str(tile) / f"{band}_{kind}.fits"


def download_mosaics(cfg: HetdexConfig, selected: tuple[int, ...] = ()) -> None:
    mosaic_dir = need(cfg.mosaic_dir, "mosaic_dir")
    base = cfg.s3_url.rstrip("/")
    jobs: list[tuple[str, Path]] = []
    for tile in native_tiles(cfg, selected):
        for (band, kind), key in mosaic_keys(cfg, tile).items():
            jobs.append((f"{base}/{key}", mosaic_path(mosaic_dir, tile, band, kind)))
    logger.info("%d mosaics", len(jobs))
    with ThreadPoolExecutor(cfg.download_workers) as pool:
        list(pool.map(lambda job: download(job[0], job[1], cfg.download_attempts), jobs))


def read_plane(path: Path) -> tuple[NDArray[Any], Any]:
    with fits.open(path, memmap=False, lazy_load_hdus=True) as h:
        i = 0 if h[0].header.get("NAXIS", 0) == 2 else 1
        head = h[i].header.copy()
        offset = h.fileinfo(i)["datLoc"]
    bitpix, scale, zero = head["BITPIX"], head.get("BSCALE", 1), head.get("BZERO", 0)
    kind = "f" if bitpix < 0 else "i"
    width = abs(bitpix) // 8
    unsigned = kind == "i" and scale == 1 and zero == 2 ** (abs(bitpix) - 1)
    check(unsigned or (scale == 1 and zero == 0), "unsupported scaling")
    dtype = np.dtype(">" + kind + str(width))
    count = head["NAXIS2"] * head["NAXIS1"]
    a = np.fromfile(path, dtype=dtype, count=count, offset=offset)
    a = a.reshape(head["NAXIS2"], head["NAXIS1"])
    a = a.byteswap().view(a.dtype.newbyteorder("="))
    if unsigned:
        a = a.view(np.dtype("u" + str(width)))
        a ^= np.array(1 << (abs(bitpix) - 1), dtype=a.dtype)
    return a, head


def cut_tile(
    mosaic_dir: Path, tile: int, ra: NDArray[np.float64], dec: NDArray[np.float64]
) -> Arrays:
    n = len(ra)
    shape = (n, IMAGE_BANDS, IMAGE_SIDE, IMAGE_SIDE)
    flux = np.full(shape, np.nan, np.float32)
    rms = np.full(shape, np.nan, np.float32)
    valid = np.zeros(shape, bool)
    flags = np.zeros(shape, np.uint32)
    present = np.zeros((n, IMAGE_BANDS), bool)
    for b, band in enumerate(MOSAIC_BANDS):
        sci, sh = read_plane(mosaic_path(mosaic_dir, tile, band, "science"))
        err, rh = read_plane(mosaic_path(mosaic_dir, tile, band, "rms"))
        flg, fh = read_plane(mosaic_path(mosaic_dir, tile, band, "flag"))
        check(
            all(rh.get(k) == sh.get(k) and fh.get(k) == sh.get(k) for k in GEOMETRY_KEYS),
            f"{band} aux geometry",
        )
        check(rh.get("FILTER") == sh.get("FILTER") == fh.get("FILTER"), "science/aux filter")
        actual = str(sh.get("FILTER", "")).upper().replace("-", "_")
        check(actual in {band, EXPECTED_FILTER[band], "NISP_" + band}, "unexpected band")
        check(flg.dtype.kind in "iu", "native flag map is not integer")
        zp = float(sh["MAGZERO"])
        check(
            np.isfinite(zp) and np.isclose(float(rh["MAGZERO"]), zp, atol=1e-8, rtol=0),
            "RMS calibration differs",
        )
        factor = 10 ** ((AB_ZERO_POINT - zp) / 2.5)
        w = WCS(sh)
        for j in range(n):
            x, y = map(float, w.world_to_pixel_values(float(ra[j]), float(dec[j])))
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            xi, yi = np.rint([x, y]).astype(int)
            inside_x = HALF <= xi <= sci.shape[1] - HALF
            inside_y = HALF <= yi <= sci.shape[0] - HALF
            if not (inside_x and inside_y):
                continue
            s = np.s_[yi - HALF : yi + HALF, xi - HALF : xi + HALF]
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                f = (np.asarray(sci[s], dtype=np.float64) * factor).astype(np.float32)
                e = (np.asarray(err[s], dtype=np.float64) * factor).astype(np.float32)
            bits = np.asarray(flg[s]).astype(np.uint32)
            flux[j, b], rms[j, b], flags[j, b] = f, e, bits
            valid[j, b] = np.isfinite(f) & np.isfinite(e) & (e > 0) & ((bits & 1) == 0)
            present[j, b] = True
        del sci, err, flg
    return dict(
        image_flux_uJy=flux,
        image_rms_uJy=rms,
        image_valid=valid,
        image_flags=flags,
        image_band_present=present,
    )


def stamp_path(stamp_dir: Path, tile: int) -> Path:
    return stamp_dir / f"{tile}.npz"


def cut_stamps(cfg: HetdexConfig, selected: tuple[int, ...] = ()) -> None:
    mosaic_dir = need(cfg.mosaic_dir, "mosaic_dir")
    stamp_dir = need(cfg.stamp_dir, "stamp_dir")
    stamp_dir.mkdir(parents=True, exist_ok=True)
    with np.load(need(cfg.native, "native")) as z:
        oid, tile_of, ra, dec = z["object_id"], z["tile"], z["ra_deg"], z["dec_deg"]
    for tile in native_tiles(cfg, selected):
        dst = stamp_path(stamp_dir, tile)
        if dst.exists():
            logger.info("tile %d: stamps exist", tile)
            continue
        rows = np.flatnonzero(tile_of == tile)
        arrays = cut_tile(mosaic_dir, tile, ra[rows], dec[rows])
        tmp = dst.with_suffix(".part")
        with tmp.open("wb") as f:
            np.savez(f, object_id=oid[rows], **cast(dict[str, Any], arrays))
        tmp.rename(dst)
        present = arrays["image_band_present"].any(1)
        logger.info("tile %d: %d objects, %d with an image", tile, len(rows), int(present.sum()))


def load_stamps(stamp_dir: Path, tile: int) -> Arrays:
    with np.load(stamp_path(stamp_dir, tile)) as z:
        return {k: z[k] for k in z.files}


def fill_packet(
    h: h5py.File,
    ids: NDArray[np.int64],
    native: Arrays,
    state: Normalizers,
    stamp_dir: Path,
) -> NDArray[np.bool_]:
    pos = {int(o): i for i, o in enumerate(native["object_id"])}
    j = np.array([pos[int(o)] for o in ids])
    check(np.all(native["field_code"][j] == 0), "N only")
    flux, err, valid = native["flux"][j], native["error"][j], native["valid"][j]
    tiles = native["tile"][j].astype(np.int64)
    n = len(ids)
    chunk = min(256, n)
    h.create_dataset("object_id", data=ids)
    h.create_dataset("field_code", data=native["field_code"][j].astype(np.uint8))
    h.create_dataset("owner_tile", data=tiles)
    h.create_dataset("table_mask", data=valid, chunks=(chunk, TABLE_SLOTS))
    h.create_dataset(
        "table_values",
        data=transform_table(flux, err, valid, state.table),
        chunks=(chunk, TABLE_SLOTS, 2),
    )
    h.create_dataset("table_available", data=valid.sum(1) >= 2)
    h.create_dataset(
        "image",
        shape=(n, 3 * IMAGE_BANDS, IMAGE_SIDE, IMAGE_SIDE),
        dtype="f4",
        chunks=(1, 3 * IMAGE_BANDS, IMAGE_SIDE, IMAGE_SIDE),
        compression="lzf",
    )
    h.create_dataset(
        "spectrum",
        data=np.zeros((n, SPECTRUM_LENGTH, 4), np.float32),
        chunks=(chunk, SPECTRUM_LENGTH, 4),
        compression="lzf",
    )
    h.create_dataset(
        "spectrum_mask",
        data=np.zeros((n, SPECTRUM_LENGTH), bool),
        chunks=(chunk, SPECTRUM_LENGTH),
        compression="lzf",
    )
    h.create_dataset("spectrum_metadata", data=np.zeros((n, 3), np.float32))
    h.create_dataset("spectrum_available", data=np.zeros(n, bool))
    image_available = np.zeros(n, bool)
    for tile in np.unique(tiles):
        stamps = load_stamps(stamp_dir, int(tile))
        at = {int(o): k for k, o in enumerate(stamps["object_id"])}
        for i in np.flatnonzero(tiles == tile):
            k = at[int(ids[i])]
            a = {key: stamps[key][k] for key in IMAGE_KEYS}
            image_available[i] = image_usable(a)
            h["image"][i] = transform_image(a, state.image)
    h.create_dataset("image_available", data=image_available)
    h.attrs["bands_json"] = json.dumps(native["bands"].tolist())
    return image_available


def packet_ids(cfg: HetdexConfig, native: Arrays) -> NDArray[np.int64]:
    want = np.load(need(cfg.build_dir, "build_dir") / IDS_FILE).astype(np.int64)
    check(np.isin(want, native["object_id"]).all(), "native photometry covers every id")
    out: NDArray[np.int64] = np.sort(want[~np.isin(want, np.array(cfg.exclude_ids, np.int64))])
    return out


def build_packet(cfg: HetdexConfig) -> None:
    out = need(cfg.packet, "packet")
    check(not out.exists(), f"{out} exists")
    state = load_normalizers(need(cfg.normalizers, "normalizers"))
    with np.load(need(cfg.native, "native"), allow_pickle=False) as z:
        native = {k: z[k] for k in z.files}
    validate_band_order(native["bands"])
    ids = packet_ids(cfg, native)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w", rdcc_nbytes=16_000_000) as h:
        image_available = fill_packet(h, ids, native, state, need(cfg.stamp_dir, "stamp_dir"))
        table_available = h["table_available"][:]
    logger.info(
        "packet %s: %d objects (%d excluded), image available %d, table available %d",
        out,
        len(ids),
        len(cfg.exclude_ids),
        int(image_available.sum()),
        int(table_available.sum()),
    )


def write_phosphoros_request(cfg: HetdexConfig) -> None:
    out = need(cfg.phosphoros_request, "phosphoros_request")
    ids = np.load(need(cfg.build_dir, "build_dir") / IDS_FILE).astype(np.int64)
    with np.load(need(cfg.native, "native")) as z:
        oid, tile, field_code = z["object_id"], z["tile"], z["field_code"]
    check(np.array_equal(oid, ids) and (field_code == 0).all(), "native rows = ids, EDF-N")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as f:
        np.savez(
            f,
            object_id=ids,
            tile=tile.astype(np.int64),
            field_code=np.zeros(len(ids), np.uint8),
        )
    logger.info("phosphoros request: %d objects -> %s", len(ids), out)


CONFIG = click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
TILES = click.option("--tile", "tiles", type=int, multiple=True)


@click.group()
def cli() -> None:
    pass


@cli.command()
@CONFIG
def select(config_path: Path) -> None:
    write_selection(load_config(config_path))


@cli.command()
@CONFIG
def native(config_path: Path) -> None:
    write_native(load_config(config_path))


@cli.command()
@CONFIG
@TILES
def mosaics(config_path: Path, tiles: tuple[int, ...]) -> None:
    download_mosaics(load_config(config_path), tiles)


@cli.command()
@CONFIG
@TILES
def cut(config_path: Path, tiles: tuple[int, ...]) -> None:
    cut_stamps(load_config(config_path), tiles)


@cli.command()
@CONFIG
def packet(config_path: Path) -> None:
    build_packet(load_config(config_path))


@cli.command("phosphoros-request")
@CONFIG
def phosphoros_request(config_path: Path) -> None:
    write_phosphoros_request(load_config(config_path))


if __name__ == "__main__":
    cli()
