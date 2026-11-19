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

from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as
from photoz.data.euclid_packet import Packet, holdout_tiles
from photoz.data.euclid_transforms import IMAGE_BANDS, IMAGE_SIDE

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
IMAGE_SHAPE = (3 * IMAGE_BANDS, IMAGE_SIDE, IMAGE_SIDE)


class PretrainStoreConfig(BaseModel):
    keys: Path
    shards_dir: Path
    store_dir: Path
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


CONFIG = click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)


@click.command()
@CONFIG
def main(config_path: Path) -> None:
    write_index(load_yaml_as(str(config_path), PretrainStoreConfig))


if __name__ == "__main__":
    main()
