from __future__ import annotations

import json
import os
import struct
import zipfile
from pathlib import Path
from typing import Any

import click
import h5py
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from lib.dataloaders.euclid_packet import KEYS, Packet, holdout_tiles
from lib.dataloaders.euclid_transforms import IMAGE_BANDS, IMAGE_SIDE
from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as

logger = get_logger()

ROW_COLUMNS = (
    "object_id",
    "selection_row",
    "field_code",
    "tile",
    "image_available",
    "image_complete",
    "image_band_present",
    "table_values",
    "table_mask",
    "table_available",
)
OLD_KEYS = tuple(k for k in KEYS if k != "image")
IMAGE_SHAPE = (3 * IMAGE_BANDS, IMAGE_SIDE, IMAGE_SIDE)
COPY_ROWS = 2048


class PretrainStoreConfig(BaseModel):
    keys: Path
    shards_dir: Path
    store_dir: Path
    stage_dir: Path
    pilot_packet: Path
    extra_packets: list[Path]
    ho_packet: Path


def stored_member(path: Path, name: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name + ".npy")
    check(info.compress_type == zipfile.ZIP_STORED, f"{name} stored uncompressed in {path}")
    with path.open("rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        check(h[:4] == b"PK\x03\x04", f"zip local header {path}")
        n, m = struct.unpack("<HH", h[26:30])
        f.seek(info.header_offset + 30 + n + m)
        version = np.lib.format.read_magic(f)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran, dtype = reader(f)
        offset = f.tell()
    check(not fortran, f"C-ordered {name} in {path}")
    return dict(offset=int(offset), shape=list(shape), dtype=dtype.str)


def exclusion_sources(
    cfg: PretrainStoreConfig,
) -> tuple[list[int], NDArray[np.integer], NDArray[np.int64], NDArray[np.int64]]:
    packet = Packet(cfg.pilot_packet)
    _, _, held_tiles, owner = holdout_tiles(packet)
    fvalid_tiles = np.unique(owner[packet.indices["valid"]])
    old = [packet.ids]
    packet.close()
    for p in cfg.extra_packets:
        with h5py.File(p, "r") as h:
            old.append(h["object_id"][:])
    with h5py.File(cfg.ho_packet, "r") as h:
        ho = h["object_id"][:]
    return held_tiles, fvalid_tiles, np.concatenate(old), ho


def build_index(cfg: PretrainStoreConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    keys = cfg.keys.read_text().split()
    cols: dict[str, list[NDArray[Any]]] = {c: [] for c in ROW_COLUMNS}
    shards, shard_index, local_index = [], [], []
    for i, k in enumerate(keys):
        path = cfg.shards_dir / f"shard_{k}.npz"
        member = stored_member(path, "image")
        with np.load(path) as z:
            for c in ROW_COLUMNS:
                cols[c].append(z[c])
            nrow = len(z["object_id"])
        check(
            member["shape"] == [nrow, *IMAGE_SHAPE] and member["dtype"] == "<f2",
            f"float16 image member {path}",
        )
        shards.append(dict(key=k, path=os.path.relpath(path, cfg.store_dir), **member))
        shard_index.append(np.full(nrow, i, np.int32))
        local_index.append(np.arange(nrow, dtype=np.int64))
    rows = {c: np.concatenate(v) for c, v in cols.items()}
    n = len(rows["object_id"])
    check(
        len(np.unique(rows["object_id"])) == n and np.isin(rows["field_code"], [1, 2]).all(),
        "unique F/S store rows",
    )
    held_tiles, fvalid_tiles, old_ids, ho = exclusion_sources(cfg)
    ids, tile = rows["object_id"], rows["tile"]
    drop = dict(
        fvalid_tile=np.isin(tile, fvalid_tiles),
        heldout_tile=np.isin(tile, held_tiles),
        ho_object=np.isin(ids, ho),
        already_in_old_sources=np.isin(ids, old_ids),
        no_image=~rows["image_available"],
    )
    keep = ~np.any(np.stack(list(drop.values())), 0)
    logger.info(
        "store: %d rows from %d owners, image_complete %d, table_available %d, "
        "dropped for ssl %s, ssl_keep %d, by field %s",
        n,
        len(keys),
        int(rows["image_complete"].sum()),
        int(rows["table_available"].sum()),
        {k: int(v.sum()) for k, v in drop.items()},
        int(keep.sum()),
        {f: int((rows["field_code"] == f).sum()) for f in (1, 2)},
    )
    index: dict[str, Any] = dict(
        ssl_keep=keep,
        shard=np.concatenate(shard_index),
        local=np.concatenate(local_index),
        **rows,
    )
    return index, shards


def write_index(cfg: PretrainStoreConfig) -> None:
    index, shards = build_index(cfg)
    cfg.store_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cfg.store_dir / "rows.npz", **index)
    (cfg.store_dir / "shards.json").write_text(json.dumps(shards, indent=1) + "\n")


def old_rows(cfg: PretrainStoreConfig) -> list[tuple[Path, NDArray[np.int64]]]:
    packet = Packet(cfg.pilot_packet)
    ssl_fit, _, held_tiles, owner = holdout_tiles(packet)
    fvalid_tiles = np.unique(owner[packet.indices["valid"]])
    packet.close()
    sources = [(cfg.pilot_packet, np.sort(ssl_fit))]
    for p in cfg.extra_packets:
        with h5py.File(p, "r") as h:
            xo = h["owner_tile"][:]
        sources.append(
            (p, np.flatnonzero(~(np.isin(xo, held_tiles) | np.isin(xo, fvalid_tiles))))
        )
    return sources


def stage_old(cfg: PretrainStoreConfig) -> None:
    sources = old_rows(cfg)
    n = sum(len(r) for _, r in sources)
    cfg.stage_dir.mkdir(parents=True, exist_ok=True)
    img = np.lib.format.open_memmap(
        cfg.stage_dir / "old_images_f16.npy", mode="w+", dtype=np.float16, shape=(n, *IMAGE_SHAPE)
    )
    arr: dict[str, Any] = {}
    ids, fields, o = [], [], 0
    for path, rows in sources:
        breaks = np.flatnonzero(np.diff(rows) != 1) + 1
        runs = list(zip(np.r_[0, breaks], np.r_[breaks, len(rows)], strict=True))
        with h5py.File(path, "r") as h:
            ids.append(h["object_id"][:][rows])
            fields.append(h["field_code"][:][rows])
            for k in OLD_KEYS:
                ds = h[k]
                if k not in arr:
                    arr[k] = np.empty((n, *ds.shape[1:]), ds.dtype)
                for s0, e0 in runs:
                    arr[k][o + s0 : o + e0] = ds[rows[s0] : rows[e0 - 1] + 1]
            ds = h["image"]
            for s0, e0 in runs:
                for c in range(s0, e0, COPY_ROWS):
                    c1 = min(e0, c + COPY_ROWS)
                    img[o + c : o + c1] = ds[rows[c] : rows[c1 - 1] + 1].astype(np.float16)
        o += len(rows)
        logger.info("staged %d rows from %s", len(rows), path)
    img.flush()
    del img
    object_id, field_code = np.concatenate(ids), np.concatenate(fields)
    check(
        o == n and np.isin(field_code, [1, 2]).all() and len(np.unique(object_id)) == n,
        "unique F/S old rows",
    )
    np.savez(cfg.stage_dir / "old_arrays.npz", object_id=object_id, field_code=field_code, **arr)
    logger.info("staged %d old rows -> %s", n, cfg.stage_dir)


CONFIG = click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)


@click.group()
def cli() -> None:
    pass


@cli.command("index")
@CONFIG
def index_command(config_path: Path) -> None:
    write_index(load_yaml_as(str(config_path), PretrainStoreConfig))


@cli.command("stage-old")
@CONFIG
def stage_old_command(config_path: Path) -> None:
    stage_old(load_yaml_as(str(config_path), PretrainStoreConfig))


if __name__ == "__main__":
    cli()
