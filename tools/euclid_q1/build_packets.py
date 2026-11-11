from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import click
import h5py
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from lib.dataloaders.euclid_transforms import (
    BANDS,
    IMAGE_SIDE,
    SPECTRUM_LENGTH,
    Normalizers,
    TableNormalizer,
    fit_normalizers,
    image_complete,
    image_usable,
    load_normalizers,
    save_normalizers,
    transform_image,
    transform_spectrum,
    transform_table,
    validate_band_order,
)
from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as

logger = get_logger()

CACHE_BYTES = 16_000_000
IMAGE_CHANNELS = 12
CHUNK_ROWS = 256
ROLE_BITS = {"SSL": 1, "FTRAIN": 2, "FVALID": 4, "NTEST": 8}
ROLE_SPLITS = (("train", "FTRAIN"), ("valid", "FVALID"), ("test", "NTEST"), ("ssl", "SSL"))
RAW14_KEYS = (
    "object_id",
    "field_code",
    "owner_tile",
    "role_bitmask",
    "flux_ujy",
    "fluxerr_ujy",
    "valid_measurement",
    "bands",
)
STAMP_GLOBS = ["*/*/products/*/*.npz", "*/products/*/*.npz"]

type Arrays = dict[str, NDArray[Any]]
type RowSpec = tuple[tuple[int, ...], str, tuple[int, ...] | None]


class LabelledPacketConfig(BaseModel):
    products_manifest: Path
    products_root: Path
    raw14: Path
    descriptor: Path
    fitting_labels: Path
    packet: Path
    normalizers: Path


class SslPacketConfig(BaseModel):
    raw14: Path
    normalizers: Path
    packet: Path
    products_root: Path
    products_manifest: Path | None = None
    registries: list[Path] = []
    product_globs: list[str] = []


class StampPacketConfig(BaseModel):
    products_root: Path
    registry_glob: str
    native: Path
    normalizers: Path
    packet: Path
    product_globs: list[str] = STAMP_GLOBS


def load_product(path: Path) -> Arrays:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def manifest_products(manifest: Path, root: Path) -> dict[int, Path]:
    entries = json.loads(manifest.read_text())["objects"]
    found = {int(x["object_id"]): root / x["path"] for x in entries}
    check(len(found) == len(entries), "unique manifest object ids")
    return found


def globbed_products(root: Path, patterns: Sequence[str]) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.with_suffix(".json").exists() and int(path.stem) not in found:
                found[int(path.stem)] = path
    return found


def read_npz(path: Path, keys: Sequence[str]) -> Arrays:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def role_indices(bits: NDArray[np.integer]) -> dict[str, NDArray[np.int64]]:
    return {name: np.flatnonzero(bits & ROLE_BITS[key]) for name, key in ROLE_SPLITS}


def fitting_targets(
    path: Path, ids: NDArray[np.int64], train: NDArray[np.int64], valid: NDArray[np.int64]
) -> Arrays:
    with np.load(path, allow_pickle=False) as z:
        fids, field, targets = z["object_id"], z["field_code"], z["z"]
        tier, source = z["label_tier"], z["label_source"]
    check(np.all(field == 1), "only F targets")
    lookup = {int(v): i for i, v in enumerate(fids)}
    names, codes = np.unique(source.astype(str), return_inverse=True)
    out: Arrays = {"label_source_lookup": names}
    for split, rows in (("train", train), ("valid", valid)):
        at = np.array([lookup[int(v)] for v in ids[rows]], np.int64)
        out[f"{split}_z"] = targets[at].astype(np.float64)
        out[f"{split}_label_tier"] = tier[at].astype(np.uint8)
        out[f"{split}_label_source_code"] = codes[at].astype(np.uint16)
    check(
        np.isfinite(out["train_z"]).all() and np.isfinite(out["valid_z"]).all(), "finite F targets"
    )
    return out


def row_specs(n: int, complete: bool) -> dict[str, RowSpec]:
    image = (n, IMAGE_CHANNELS, IMAGE_SIDE, IMAGE_SIDE)
    specs: dict[str, RowSpec] = {
        "image": (image, "f4", (1, IMAGE_CHANNELS, IMAGE_SIDE, IMAGE_SIDE)),
        "image_available": ((n,), "?", None),
    }
    if complete:
        specs["image_complete"] = ((n,), "?", None)
    specs |= {
        "spectrum": ((n, SPECTRUM_LENGTH, 4), "f4", (1, SPECTRUM_LENGTH, 4)),
        "spectrum_mask": ((n, SPECTRUM_LENGTH), "?", (1, SPECTRUM_LENGTH)),
        "spectrum_metadata": ((n, 3), "f4", None),
        "spectrum_available": ((n,), "?", None),
    }
    return specs


def create_rows(h: h5py.File, specs: Mapping[str, RowSpec]) -> None:
    for key, (shape, dtype, chunks) in specs.items():
        h.create_dataset(
            key, shape=shape, dtype=dtype, chunks=chunks, compression="lzf" if chunks else None
        )


def write_table(
    h: h5py.File,
    flux: NDArray[np.floating],
    error: NDArray[np.floating],
    valid: NDArray[np.bool_],
    state: TableNormalizer,
) -> None:
    chunk = min(CHUNK_ROWS, len(valid))
    h.create_dataset("table_mask", data=valid, chunks=(chunk, len(BANDS)))
    h.create_dataset(
        "table_values",
        data=transform_table(flux, error, valid, state),
        chunks=(chunk, len(BANDS), 2),
    )
    h.create_dataset("table_available", data=valid.sum(1) >= 2)


def write_spectrum(h: h5py.File, i: int, a: Mapping[str, NDArray[Any]], floor: float) -> bool:
    values, mask, metadata, available = transform_spectrum(a, floor)
    h["spectrum"][i] = values
    h["spectrum_mask"][i] = mask
    h["spectrum_metadata"][i] = metadata
    return available


def write_labelled_rows(
    h: h5py.File, ids: NDArray[np.int64], products: Mapping[int, Path], state: Normalizers
) -> None:
    for i, identifier in enumerate(ids):
        a = load_product(products[int(identifier)])
        h["image"][i] = transform_image(a, state.image)
        h["image_available"][i] = image_usable(a)
        h["image_complete"][i] = image_complete(a)
        h["spectrum_available"][i] = write_spectrum(h, i, a, state.spectrum.floor_cgs)
        if i % 10000 == 0:
            logger.info("labelled packet rows %d/%d", i, len(ids))


def write_ssl_rows(
    h: h5py.File, ids: NDArray[np.int64], products: Mapping[int, Path], state: Normalizers
) -> None:
    image_available = np.zeros(len(ids), bool)
    spectrum_available = np.zeros(len(ids), bool)
    for i, identifier in enumerate(ids):
        a = load_product(products[int(identifier)])
        image_available[i] = image_usable(a)
        h["image"][i] = transform_image(a, state.image)
        spectrum_available[i] = write_spectrum(h, i, a, state.spectrum.floor_cgs)
        if i % 10000 == 0:
            logger.info("ssl packet rows %d/%d", i, len(ids))
    h["image_available"][:] = image_available
    h["spectrum_available"][:] = spectrum_available


def write_stamp_rows(
    h: h5py.File, ids: NDArray[np.int64], products: Mapping[int, Path], state: Normalizers
) -> None:
    image_available = np.zeros(len(ids), bool)
    for i, identifier in enumerate(ids):
        a = load_product(products[int(identifier)])
        image_available[i] = image_usable(a)
        h["image"][i] = transform_image(a, state.image)
    h.create_dataset("image_available", data=image_available)


def write_labelled_packet(
    h: h5py.File,
    raw: Arrays,
    roles: Mapping[str, NDArray[np.int64]],
    targets: Arrays,
    descriptor: NDArray[Any],
    products: Mapping[int, Path],
    state: Normalizers,
) -> None:
    n = len(raw["object_id"])
    for key in ("object_id", "field_code", "owner_tile", "role_bitmask"):
        h.create_dataset(key, data=raw[key])
    for split in ("train", "valid", "test", "ssl"):
        h.create_dataset(f"{split}_indices", data=roles[split])
    h.create_dataset("train_z", data=targets["train_z"])
    h.create_dataset("valid_z", data=targets["valid_z"])
    h.create_dataset("table_descriptor", data=descriptor)
    for key in (
        "train_label_tier",
        "valid_label_tier",
        "train_label_source_code",
        "valid_label_source_code",
    ):
        h.create_dataset(key, data=targets[key])
    h.create_dataset(
        "label_source_lookup",
        data=targets["label_source_lookup"].astype(object),
        dtype=h5py.string_dtype("utf-8"),
    )
    chunk = min(CHUNK_ROWS, n)
    h.create_dataset("raw_photometry_flux_ujy", data=raw["flux_ujy"], chunks=(chunk, len(BANDS)))
    h.create_dataset(
        "raw_photometry_error_ujy", data=raw["fluxerr_ujy"], chunks=(chunk, len(BANDS))
    )
    write_table(h, raw["flux_ujy"], raw["fluxerr_ujy"], raw["valid_measurement"], state.table)
    create_rows(h, row_specs(n, complete=True))
    write_labelled_rows(h, raw["object_id"], products, state)
    h.attrs["bands_json"] = json.dumps(raw["bands"].tolist())
    h.attrs["role_bits_json"] = json.dumps(ROLE_BITS)


def build_labelled(config: LabelledPacketConfig) -> None:
    products = manifest_products(config.products_manifest, config.products_root)
    raw = read_npz(config.raw14, RAW14_KEYS)
    validate_band_order(raw["bands"])
    ids = raw["object_id"]
    roles = role_indices(raw["role_bitmask"])
    train = roles["train"]
    logger.info("labelled packet: %s", {k: len(v) for k, v in roles.items()} | {"rows": len(ids)})
    state = fit_normalizers(
        ids[train],
        ids[roles["ssl"]],
        lambda identifier: load_product(products[identifier]),
        raw["flux_ujy"][train],
        raw["fluxerr_ujy"][train],
        raw["valid_measurement"][train],
    )
    descriptor = np.load(config.descriptor, allow_pickle=False)
    check(descriptor.shape == (len(BANDS), 18) and np.isfinite(descriptor).all(), "descriptors")
    targets = fitting_targets(config.fitting_labels, ids, train, roles["valid"])
    config.packet.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(config.packet, "w-", rdcc_nbytes=CACHE_BYTES) as h:
        write_labelled_packet(h, raw, roles, targets, descriptor, products, state)
    save_normalizers(state, config.normalizers)
    logger.info("wrote %s and %s", config.packet, config.normalizers)


def ssl_products(config: SslPacketConfig) -> tuple[NDArray[np.int64], dict[int, Path]]:
    if config.products_manifest is not None:
        products = manifest_products(config.products_manifest, config.products_root)
        return np.array(sorted(products), np.int64), products
    products = globbed_products(config.products_root, config.product_globs)
    expected = [
        int(r["object_id"]) for path in config.registries for r in json.loads(path.read_text())
    ]
    missing = [o for o in expected if o not in products]
    check(not missing, f"missing products {len(missing)}")
    return np.array(sorted(expected), np.int64), products


def write_ssl_packet(
    h: h5py.File,
    ids: NDArray[np.int64],
    raw: Arrays,
    rows: NDArray[np.int64],
    products: Mapping[int, Path],
    state: Normalizers,
) -> None:
    field, bits = raw["field_code"][rows], raw["role_bitmask"][rows]
    check(np.isin(field, (1, 2)).all() and np.all(bits == ROLE_BITS["SSL"]), "F/S SSL rows")
    h.create_dataset("object_id", data=ids)
    h.create_dataset("field_code", data=field)
    h.create_dataset("role_bitmask", data=bits)
    h.create_dataset("owner_tile", data=raw["owner_tile"][rows])
    write_table(
        h,
        raw["flux_ujy"][rows],
        raw["fluxerr_ujy"][rows],
        raw["valid_measurement"][rows],
        state.table,
    )
    create_rows(h, row_specs(len(ids), complete=False))
    write_ssl_rows(h, ids, products, state)


def build_ssl(config: SslPacketConfig) -> None:
    state = load_normalizers(config.normalizers)
    ids, products = ssl_products(config)
    raw = read_npz(config.raw14, RAW14_KEYS)
    validate_band_order(raw["bands"])
    pos = {int(o): i for i, o in enumerate(raw["object_id"])}
    rows = np.array([pos[int(o)] for o in ids], np.int64)
    logger.info("ssl packet rows %d", len(ids))
    config.packet.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(config.packet, "w-", rdcc_nbytes=CACHE_BYTES) as h:
        write_ssl_packet(h, ids, raw, rows, products, state)
    logger.info("wrote %s", config.packet)


def stamp_rows(config: StampPacketConfig) -> list[dict[str, Any]]:
    return [
        r
        for path in sorted(config.products_root.glob(config.registry_glob))
        for r in json.loads(path.read_text())
    ]


def write_stamp_packet(
    h: h5py.File,
    ids: NDArray[np.int64],
    owner: Mapping[int, int],
    native: Arrays,
    products: Mapping[int, Path],
    state: Normalizers,
) -> None:
    pos = {int(o): i for i, o in enumerate(native["object_id"])}
    j = np.array([pos[int(o)] for o in ids], np.int64)
    n = len(ids)
    chunk = min(CHUNK_ROWS, n)
    h.create_dataset("object_id", data=ids)
    h.create_dataset("field_code", data=native["field_code"][j].astype(np.uint8))
    h.create_dataset("owner_tile", data=np.array([owner[int(o)] for o in ids], np.int64))
    write_table(h, native["flux"][j], native["error"][j], native["valid"][j], state.table)
    image = (n, IMAGE_CHANNELS, IMAGE_SIDE, IMAGE_SIDE)
    h.create_dataset("image", shape=image, dtype="f4", chunks=(1, *image[1:]), compression="lzf")
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
    write_stamp_rows(h, ids, products, state)
    h.attrs["bands_json"] = json.dumps(native["bands"].tolist())


def build_stamps(config: StampPacketConfig) -> None:
    state = load_normalizers(config.normalizers)
    rows = stamp_rows(config)
    products = globbed_products(config.products_root, config.product_globs)
    ids = np.array(
        sorted(int(r["object_id"]) for r in rows if int(r["object_id"]) in products), np.int64
    )
    owner = {int(r["object_id"]): int(r["owner_tile"]) for r in rows}
    native = read_npz(config.native, ("object_id", "field_code", "flux", "error", "valid", "bands"))
    validate_band_order(native["bands"])
    logger.info("stamp packet rows %d (registry %d)", len(ids), len(rows))
    config.packet.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(config.packet, "w-", rdcc_nbytes=CACHE_BYTES) as h:
        write_stamp_packet(h, ids, owner, native, products, state)
    logger.info("wrote %s", config.packet)


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
def labelled(config_path: Path) -> None:
    build_labelled(load_yaml_as(str(config_path), LabelledPacketConfig))


@cli.command()
@CONFIG
def ssl(config_path: Path) -> None:
    build_ssl(load_yaml_as(str(config_path), SslPacketConfig))


@cli.command()
@CONFIG
def stamps(config_path: Path) -> None:
    build_stamps(load_yaml_as(str(config_path), StampPacketConfig))


if __name__ == "__main__":
    cli()
