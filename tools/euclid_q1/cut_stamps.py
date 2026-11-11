from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import click
import numpy as np
import pyarrow.parquet as pq
from astropy.io import fits
from astropy.wcs import WCS
from numpy.typing import NDArray
from pydantic import BaseModel

from lib.dataloaders.euclid_transforms import (
    BANDS,
    IMAGE_BANDS,
    IMAGE_SIDE,
    ImageNormalizer,
    load_normalizers,
    transform_image_band,
    transform_table,
)
from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as

logger = get_logger()

IMAGE_BAND_NAMES = ("VIS", "Y", "J", "H")
KINDS = ("science", "rms", "flag")
HALF = IMAGE_SIDE // 2
AB_ZEROPOINT_UJY = 23.9
GEOMETRY = (
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

type Plane = NDArray[Any] | SparsePlane


class CutStampsConfig(BaseModel):
    selected: Path
    raw14_dir: Path
    raw14_glob: str = "raw14_*.npz"
    mosaic_dirs: list[str]
    normalizers: Path
    inputs_dir: Path
    owners: Path
    keys: Path
    shards_dir: Path
    read_flags: bool = False
    sparse_rows: int = 1000
    workers: int = 1


def owner_key(field_code: int, tile: int) -> str:
    return f"{'F' if field_code == 1 else 'S'}{tile}"


def mosaic_file(templates: list[str], field_code: int, tile: int, band: str, kind: str) -> Path:
    key = owner_key(field_code, tile)
    name = f"{tile}_{band}_{kind}.fits"
    candidates = [Path(t.format(tile=tile, key=key)) / name for t in templates]
    return next((c for c in candidates if c.is_file()), candidates[0])


def load_raw14(cfg: CutStampsConfig, object_id: NDArray[np.int64]) -> dict[str, NDArray[Any]]:
    n = len(object_id)
    flux = np.full((n, len(BANDS)), np.nan, np.float32)
    error = np.full((n, len(BANDS)), np.nan, np.float32)
    valid = np.zeros((n, len(BANDS)), bool)
    seen = np.zeros(n, np.int32)
    bands: NDArray[Any] | None = None
    for path in sorted(cfg.raw14_dir.glob(cfg.raw14_glob)):
        with np.load(path) as z:
            s = z["selection_row"]
            check(np.array_equal(z["object_id"], object_id[s]), f"raw14 object ids {path}")
            flux[s], error[s], valid[s] = z["flux_ujy"], z["fluxerr_ujy"], z["valid_measurement"]
            seen[s] += 1
            if bands is None:
                bands = z["bands"]
            check(np.array_equal(bands, z["bands"]), f"raw14 band order {path}")
    if bands is None or not (seen == 1).all():
        raise ValueError("raw14 must cover every selected row exactly once")
    return dict(flux=flux, err=error, valid=valid, bands=bands)


def owner_inputs(cfg: CutStampsConfig) -> list[tuple[dict[str, Any], dict[str, NDArray[Any]]]]:
    table = pq.read_table(
        cfg.selected, columns=["object_id", "field_code", "tile", "ra_deg", "dec_deg"]
    )
    t = {k: table[k].to_numpy() for k in table.column_names}
    n = len(t["object_id"])
    raw = load_raw14(cfg, t["object_id"])
    order = np.lexsort((np.arange(n), t["tile"], t["field_code"]))
    groups = np.flatnonzero(
        np.diff(t["field_code"][order].astype(np.int64)) | np.diff(t["tile"][order])
    )
    owners = []
    for idx in np.split(order, groups + 1):
        fc, tile = int(t["field_code"][idx[0]]), int(t["tile"][idx[0]])
        files = {}
        for band in IMAGE_BAND_NAMES:
            for kind in KINDS:
                path = mosaic_file(cfg.mosaic_dirs, fc, tile, band, kind)
                check(path.is_file(), f"missing mosaic {path}")
                files[f"{band}_{kind}"] = str(path)
        arrays = dict(
            object_id=t["object_id"][idx],
            field_code=np.full(len(idx), fc, np.uint8),
            tile=np.full(len(idx), tile, np.int64),
            ra_deg=t["ra_deg"][idx],
            dec_deg=t["dec_deg"][idx],
            selection_row=idx.astype(np.int64),
            flux=raw["flux"][idx],
            err=raw["err"][idx],
            valid=raw["valid"][idx],
            bands=raw["bands"],
        )
        owner = dict(key=owner_key(fc, tile), field_code=fc, tile=tile, rows=len(idx), files=files)
        owners.append((owner, arrays))
    return owners


def key_order(owners: list[dict[str, Any]]) -> list[str]:
    return [str(o["key"]) for o in sorted(owners, key=lambda o: -int(o["rows"]))]


def prepare_inputs(cfg: CutStampsConfig) -> None:
    prepared = owner_inputs(cfg)
    cfg.inputs_dir.mkdir(parents=True, exist_ok=True)
    for owner, arrays in prepared:
        payload: dict[str, Any] = arrays
        np.savez(cfg.inputs_dir / f"owner_{owner['key']}.npz", **payload)
    owners = [owner for owner, _ in prepared]
    cfg.owners.parent.mkdir(parents=True, exist_ok=True)
    cfg.owners.write_text(json.dumps(owners, indent=1) + "\n")
    cfg.keys.parent.mkdir(parents=True, exist_ok=True)
    cfg.keys.write_text("\n".join(key_order(owners)) + "\n")
    logger.info(
        "prepared %d rows in %d owners", sum(int(o["rows"]) for o in owners), len(owners)
    )


def image_header(path: str) -> tuple[fits.Header, int, np.dtype[Any], bool]:
    with fits.open(path, memmap=False, lazy_load_hdus=True) as h:
        i = 0 if h[0].header.get("NAXIS", 0) == 2 else 1
        head = h[i].header.copy()
        offset = int(h.fileinfo(i)["datLoc"])
    bitpix = int(head["BITPIX"])
    scale, zero = head.get("BSCALE", 1), head.get("BZERO", 0)
    kind = "f" if bitpix < 0 else "i"
    unsigned = kind == "i" and scale == 1 and zero == 2 ** (abs(bitpix) - 1)
    check(unsigned or (scale == 1 and zero == 0), f"unsupported scaling {path}")
    return head, offset, np.dtype(">" + kind + str(abs(bitpix) // 8)), unsigned


def native(data: NDArray[Any], unsigned: bool, bits: int) -> NDArray[Any]:
    data = data.byteswap().view(data.dtype.newbyteorder("="))
    if unsigned:
        data = data.view(np.dtype("u" + str(bits // 8)))
        data ^= np.array(1 << (bits - 1), dtype=data.dtype)
    return data


def load_plane(path: str) -> tuple[NDArray[Any], fits.Header]:
    head, offset, dtype, unsigned = image_header(path)
    data = np.fromfile(
        path, dtype=dtype, count=head["NAXIS2"] * head["NAXIS1"], offset=offset
    ).reshape(head["NAXIS2"], head["NAXIS1"])
    return native(data, unsigned, abs(head["BITPIX"])), head


class SparsePlane:
    def __init__(self, path: str) -> None:
        self.path = path
        self.head, self.offset, self.dtype, self.unsigned = image_header(path)
        self.shape = (int(self.head["NAXIS2"]), int(self.head["NAXIS1"]))

    def stamp(self, y0: int, x0: int) -> NDArray[Any]:
        w = self.shape[1]
        block = np.fromfile(
            self.path,
            dtype=self.dtype,
            count=IMAGE_SIDE * w,
            offset=self.offset + y0 * w * self.dtype.itemsize,
        ).reshape(IMAGE_SIDE, w)
        return native(
            np.ascontiguousarray(block[:, x0 : x0 + IMAGE_SIDE]),
            self.unsigned,
            abs(self.head["BITPIX"]),
        )


def open_plane(path: str, n: int, sparse_rows: int) -> tuple[Plane, fits.Header]:
    if n <= sparse_rows:
        plane = SparsePlane(path)
        return plane, plane.head
    return load_plane(path)


def stamp(src: Plane, y: int, x: int) -> NDArray[Any]:
    if isinstance(src, SparsePlane):
        return src.stamp(y - HALF, x - HALF)
    return src[y - HALF : y + HALF, x - HALF : x + HALF]


def stamps(
    src: Plane,
    yi: NDArray[np.int64],
    xi: NDArray[np.int64],
    rows: NDArray[np.intp],
    dtype: type[np.generic],
) -> NDArray[Any]:
    out = np.empty((len(rows), IMAGE_SIDE, IMAGE_SIDE), dtype)
    for j, r in enumerate(rows):
        out[j] = stamp(src, int(yi[r]), int(xi[r]))
    return out


def same_geometry(a: fits.Header, b: fits.Header) -> bool:
    return all(a.get(k) == b.get(k) for k in GEOMETRY)


def cut_owner(
    files: dict[str, str],
    z: dict[str, NDArray[Any]],
    image_state: ImageNormalizer,
    read_flags: bool,
    sparse_rows: int,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    n = len(z["object_id"])
    ra = np.array([float(f"{v:.12f}") for v in z["ra_deg"]])
    dec = np.array([float(f"{v:.12f}") for v in z["dec_deg"]])
    img = np.zeros((n, 3 * IMAGE_BANDS, IMAGE_SIDE, IMAGE_SIDE), np.float16)
    present = np.zeros((n, IMAGE_BANDS), bool)
    anyvalid = np.zeros((n, IMAGE_BANDS), bool)
    stats: dict[str, Any] = {}
    for b, band in enumerate(IMAGE_BAND_NAMES):
        sci, sh = open_plane(files[f"{band}_science"], n, sparse_rows)
        x, y = WCS(sh).world_to_pixel_values(ra, dec)
        ok = np.isfinite(x) & np.isfinite(y)
        xi = np.rint(np.where(ok, x, -1e9)).astype(np.int64)
        yi = np.rint(np.where(ok, y, -1e9)).astype(np.int64)
        hgt, wid = sci.shape
        inb = ok & (xi >= HALF) & (xi <= wid - HALF) & (yi >= HALF) & (yi <= hgt - HALF)
        rows = np.flatnonzero(inb)
        science = stamps(sci, yi, xi, rows, np.float32)
        zp = float(sh["MAGZERO"])
        del sci
        rms, rh = open_plane(files[f"{band}_rms"], n, sparse_rows)
        check(same_geometry(rh, sh), f"{band} rms geometry")
        check(np.isclose(float(rh["MAGZERO"]), zp, atol=1e-8, rtol=0), f"{band} rms calibration")
        noise = stamps(rms, yi, xi, rows, np.float32)
        del rms
        flags = np.zeros((len(rows), IMAGE_SIDE, IMAGE_SIDE), np.uint32)
        if read_flags:
            flg, fh = open_plane(files[f"{band}_flag"], n, sparse_rows)
            check(same_geometry(fh, sh), f"{band} flag geometry")
            check(flg.dtype.kind in "iu", f"{band} integer flags")
            flags = stamps(flg, yi, xi, rows, np.uint32)
            del flg
        factor = 10 ** ((AB_ZEROPOINT_UJY - zp) / 2.5)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            flux = (science.astype(np.float64) * factor).astype(np.float32)
            error = (noise.astype(np.float64) * factor).astype(np.float32)
        valid = np.isfinite(flux) & np.isfinite(error) & (error > 0) & ((flags & 1) == 0)
        cf, ce = transform_image_band(flux, error, valid, image_state, b)
        img[rows, b] = cf.astype(np.float16)
        img[rows, IMAGE_BANDS + b] = ce.astype(np.float16)
        img[rows, 2 * IMAGE_BANDS + b] = valid.astype(np.float16)
        present[rows, b] = True
        anyvalid[rows, b] = valid.reshape(len(rows), IMAGE_SIDE * IMAGE_SIDE).any(1)
        stats[band] = dict(
            in_bounds=int(len(rows)),
            valid_pixel_fraction=float(valid.mean()) if len(rows) else None,
            magzero=zp,
        )
    check(np.isfinite(img.astype(np.float32)).all(), "finite float16 image")
    usable = present & anyvalid
    out = dict(
        object_id=z["object_id"],
        selection_row=z["selection_row"],
        field_code=z["field_code"],
        tile=z["tile"],
        image=img,
        image_band_present=present,
        image_available=usable.any(1),
        image_complete=usable.all(1),
        table_mask=z["valid"],
        table_available=z["valid"].sum(1) >= 2,
        flags_read=np.array(read_flags),
    )
    return out, stats


def cut_key(cfg: CutStampsConfig, key: str) -> None:
    dst = cfg.shards_dir / f"shard_{key}.npz"
    check(not dst.exists(), f"shard exists {dst}")
    owner = next(o for o in json.loads(cfg.owners.read_text()) if o["key"] == key)
    norm = load_normalizers(cfg.normalizers)
    t0 = time.time()
    with np.load(cfg.inputs_dir / f"owner_{key}.npz") as f:
        z = {k: f[k] for k in f.files}
    out, stats = cut_owner(owner["files"], z, norm.image, cfg.read_flags, cfg.sparse_rows)
    table = transform_table(
        z["flux"].astype(np.float64), z["err"].astype(np.float64), z["valid"], norm.table
    )
    out["table_values"] = table.astype(np.float32)
    order = (
        "object_id",
        "selection_row",
        "field_code",
        "tile",
        "image",
        "image_band_present",
        "image_available",
        "image_complete",
        "table_values",
        "table_mask",
        "table_available",
        "flags_read",
    )
    cfg.shards_dir.mkdir(parents=True, exist_ok=True)
    tmp = cfg.shards_dir / f"shard_{key}.partial.npz"
    arrays: dict[str, Any] = {k: out[k] for k in order}
    np.savez(tmp, **arrays)
    os.replace(tmp, dst)
    logger.info(
        "%s: %d rows in %.1f s, image_available %d, image_complete %d, table_available %d, %s",
        key,
        len(z["object_id"]),
        time.time() - t0,
        int(out["image_available"].sum()),
        int(out["image_complete"].sum()),
        int(out["table_available"].sum()),
        json.dumps(stats),
    )


CONFIG = click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)


@click.group()
def cli() -> None:
    pass


@cli.command("prepare")
@CONFIG
def prepare_command(config_path: Path) -> None:
    prepare_inputs(load_yaml_as(str(config_path), CutStampsConfig))


@cli.command("cut")
@CONFIG
@click.option("--key", "keys", multiple=True)
def cut_command(config_path: Path, keys: tuple[str, ...]) -> None:
    cfg = load_yaml_as(str(config_path), CutStampsConfig)
    todo = list(keys) or [
        k
        for k in cfg.keys.read_text().split()
        if not (cfg.shards_dir / f"shard_{k}.npz").exists()
    ]
    if cfg.workers <= 1:
        for k in todo:
            cut_key(cfg, k)
        return
    with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
        for future in [pool.submit(cut_key, cfg, k) for k in todo]:
            future.result()


if __name__ == "__main__":
    cli()
