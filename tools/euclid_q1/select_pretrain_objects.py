from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, BinaryIO

import click
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.io import fits
from numpy.typing import NDArray
from pydantic import BaseModel
from scipy.spatial import cKDTree

from lib.dataloaders.euclid_transforms import BANDS
from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as

logger = get_logger()

NULL_ID = int(np.iinfo(np.int64).max)
SOURCE_TYPES = {
    "object_id": pa.int64(),
    "field_code": pa.uint8(),
    "tile": pa.int64(),
    "source_file_index": pa.int32(),
    "source_row": pa.int64(),
    "ra_deg": pa.float64(),
    "dec_deg": pa.float64(),
    "id_hash": pa.uint64(),
}
COLUMNS = tuple(SOURCE_TYPES)
RECORD_COLUMNS = (*COLUMNS, "sample_hash")
RECORD = np.dtype(
    [
        ("object_id", "<i8"),
        ("field_code", "u1"),
        ("tile", "<i8"),
        ("source_file_index", "<i4"),
        ("source_row", "<i8"),
        ("ra_deg", "<f8"),
        ("dec_deg", "<f8"),
        ("id_hash", "<u8"),
        ("sample_hash", "<u8"),
    ]
)
SCHEMA = pa.schema(
    [pa.field(k, v) for k, v in SOURCE_TYPES.items()] + [pa.field("sample_hash", pa.uint64())]
)
FIELD_CODES = {"EDF-F": 1, "EDF-S": 2}
GUARD_BIT, NEIGHBOR_BIT, UNKNOWN_BIT = 1, 2, 4
FATAL = GUARD_BIT | NEIGHBOR_BIT | UNKNOWN_BIT
ANGLE_BAND_RAD = 1e-12
QUERY_PADDING_CHORD = 4e-12
ARCSEC_RAD = np.pi / (180.0 * 3600.0)
DECIMAL_PI = Decimal(
    "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803"
    "48253421170679"
)
GUARD_KEYS = ("blocked_tiles", "blocked_ids", "old_ids", "guard_ra_deg", "guard_dec_deg")
WRITE_ROWS = 32768
FITS_BLOCK = 2880
TFORM_SIZES = {"L": 1, "B": 1, "I": 2, "J": 4, "K": 8, "A": 1, "E": 4, "D": 8, "C": 8, "M": 16}
FLUX = tuple(f"FLUX_{b}_2FWHM_APER" for b in BANDS)
ERROR = tuple(f"FLUXERR_{b}_2FWHM_APER" for b in BANDS)
CATALOG_FORMS = {"OBJECT_ID": "K", **dict.fromkeys(FLUX + ERROR, "E")}

type Columns = dict[str, NDArray[Any]]


class SelectionConfig(BaseModel):
    registry_manifest: Path
    guards: Path
    selected: Path
    mer_catalog_dir: Path
    raw14_dir: Path
    target_per_field: int = 500_000
    reservoir_per_field: int = 1_000_000
    sample_seed: int = 20261101
    guard_radius_arcsec: float = 10.0
    neighbor_radius_arcsec: float = 1.0
    batch_rows: int = 8192
    merge_rows: int = 262_144


def splitmix64(ids: NDArray[np.int64], seed: int) -> NDArray[np.uint64]:
    a = np.asarray(ids, dtype=np.int64).view(np.uint64).copy()
    with np.errstate(over="ignore"):
        a += np.uint64(seed)
        a += np.uint64(0x9E3779B97F4A7C15)
        a = (a ^ (a >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        a = (a ^ (a >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    out: NDArray[np.uint64] = a ^ (a >> np.uint64(31))
    return out


def finite_coords(ra: NDArray[np.float64], dec: NDArray[np.float64]) -> NDArray[np.bool_]:
    out: NDArray[np.bool_] = (
        np.isfinite(ra) & np.isfinite(dec) & (ra >= 0) & (ra <= 360) & (dec >= -90) & (dec <= 90)
    )
    return out


def xyz(ra: NDArray[np.float64], dec: NDArray[np.float64]) -> NDArray[np.float64]:
    r, d = np.deg2rad(ra), np.deg2rad(dec)
    c = np.cos(d)
    return np.column_stack((c * np.cos(r), c * np.sin(r), np.sin(d)))


def angle(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = np.arctan2(
        np.linalg.norm(np.cross(a, b), axis=-1), np.clip(np.sum(a * b, axis=-1), -1, 1)
    )
    return out


def decimal_sine(x: Decimal) -> Decimal:
    x %= 2 * DECIMAL_PI
    if x > DECIMAL_PI:
        x -= 2 * DECIMAL_PI
    if x < -DECIMAL_PI:
        x += 2 * DECIMAL_PI
    total = term = x
    for n in range(1, 160):
        term *= -x * x / ((2 * n) * (2 * n + 1))
        previous = total
        total += term
        if total == previous:
            return total
    raise ArithmeticError("Decimal sine convergence")


def boundary_within(ra1: float, dec1: float, ra2: float, dec2: float, threshold: float) -> bool:
    with localcontext() as context:
        context.prec = 80
        r1, d1, r2, d2 = (Decimal.from_float(float(v)) for v in (ra1, dec1, ra2, dec2))
        unit = DECIMAL_PI / 180
        h = decimal_sine((d2 - d1) * unit / 2) ** 2
        h += (
            decimal_sine(DECIMAL_PI / 2 - d1 * unit)
            * decimal_sine(DECIMAL_PI / 2 - d2 * unit)
            * decimal_sine((r2 - r1) * unit / 2) ** 2
        )
        return bool(h <= decimal_sine(Decimal.from_float(float(threshold)) / 2) ** 2)


def contains(sorted_ids: NDArray[np.int64], values: NDArray[np.int64]) -> NDArray[np.bool_]:
    if not len(sorted_ids):
        return np.zeros(len(values), dtype=bool)
    ix = np.searchsorted(sorted_ids, values)
    out: NDArray[np.bool_] = (ix < len(sorted_ids)) & (
        sorted_ids[np.minimum(ix, len(sorted_ids) - 1)] == values
    )
    return out


def ordered(rows: NDArray[np.void]) -> NDArray[np.void]:
    return rows[np.lexsort((rows["object_id"], rows["sample_hash"], rows["field_code"]))]


class Reservoir:
    def __init__(self, keep: int, merge_rows: int) -> None:
        self.keep = keep
        self.n = 0
        self.data: NDArray[np.void] = np.empty(keep + merge_rows, dtype=RECORD)
        self.threshold: tuple[Any, Any] | None = None

    def reduce(self) -> None:
        if self.n > self.keep:
            take = np.argpartition(
                self.data[: self.n], self.keep - 1, order=["sample_hash", "object_id"]
            )[: self.keep]
            kept = self.data[: self.n][take]
            self.data[: self.keep] = kept
            self.n = self.keep
        if self.n == self.keep:
            h = self.data["sample_hash"][: self.n].max()
            oid = self.data["object_id"][: self.n][self.data["sample_hash"][: self.n] == h].max()
            self.threshold = (h, oid)

    def push(self, rows: NDArray[np.void]) -> None:
        if self.threshold is not None:
            h, oid = self.threshold
            rows = rows[
                (rows["sample_hash"] < h) | ((rows["sample_hash"] == h) & (rows["object_id"] < oid))
            ]
        pos = 0
        while pos < len(rows):
            if self.n == len(self.data):
                self.reduce()
            count = min(len(rows) - pos, len(self.data) - self.n)
            self.data[self.n : self.n + count] = rows[pos : pos + count]
            self.n += count
            pos += count

    def finish(self) -> NDArray[np.void]:
        self.reduce()
        return ordered(self.data[: self.n].copy())


def registry_plan(manifest: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    sources = {int(s["source_file_index"]): s for s in manifest["sources"]}
    plans = []
    for shard in manifest["shards"]:
        if shard["kind"] != "registry":
            continue
        match = re.fullmatch(r"registry/(\d{4})_(\d{10})\.parquet", shard["path"])
        if match is None:
            raise ValueError(f"unrecognized registry path {shard['path']}")
        index, start = int(match[1]), int(match[2])
        source = sources[index]
        if source["field"] not in FIELD_CODES:
            continue
        plans.append(
            dict(
                path=root / shard["path"],
                rows=int(shard["rows"]),
                source_index=index,
                start_row=start,
                field_code=FIELD_CODES[source["field"]],
                tile=int(source["tile"]),
            )
        )
    return sorted(plans, key=lambda s: (s["source_index"], s["start_row"]))


def load_guards(path: Path) -> Columns:
    with np.load(path, allow_pickle=False) as z:
        g = {k: z[k] for k in GUARD_KEYS}
    for k in ("blocked_tiles", "blocked_ids", "old_ids"):
        check(
            g[k].dtype == np.int64 and np.array_equal(g[k], np.unique(g[k])), f"sorted unique {k}"
        )
    n = len(g["blocked_ids"])
    check(
        g["guard_ra_deg"].shape == g["guard_dec_deg"].shape == (n,), "guard anchors aligned to ids"
    )
    check(finite_coords(g["guard_ra_deg"], g["guard_dec_deg"]).all(), "finite guard anchors")
    return g


def iter_registry(plans: list[dict[str, Any]], batch_rows: int) -> Iterator[Columns]:
    for spec in plans:
        f = pq.ParquetFile(spec["path"])
        check(f.metadata.num_rows == spec["rows"], f"registry rows {spec['path']}")
        offset = 0
        batches = f.iter_batches(batch_size=batch_rows, columns=list(COLUMNS), use_threads=False)
        for batch in batches:
            a: Columns = {}
            for k in COLUMNS:
                col = batch.column(batch.schema.get_field_index(k))
                if k == "object_id":
                    col = col.fill_null(NULL_ID)
                elif k in ("ra_deg", "dec_deg"):
                    col = col.fill_null(float("nan"))
                else:
                    check(col.null_count == 0, f"null registry column {k}")
                a[k] = col.to_numpy(zero_copy_only=False)
            n = len(a["object_id"])
            check(
                (a["field_code"] == spec["field_code"]).all()
                and (a["tile"] == spec["tile"]).all()
                and (a["source_file_index"] == spec["source_index"]).all()
                and np.array_equal(
                    a["source_row"],
                    np.arange(spec["start_row"] + offset, spec["start_row"] + offset + n),
                ),
                f"registry identity {spec['path']}",
            )
            offset += n
            yield a
        check(offset == spec["rows"], f"registry rows decoded {spec['path']}")


def metadata_keep(a: Columns, g: Columns) -> tuple[NDArray[np.bool_], dict[str, int]]:
    valid_id = a["object_id"] != NULL_ID
    valid_coord = finite_coords(a["ra_deg"], a["dec_deg"])
    owner = contains(g["blocked_tiles"], a["tile"])
    blocked = contains(g["blocked_ids"], a["object_id"])
    old = contains(g["old_ids"], a["object_id"])
    check(not np.any((owner | blocked) & ~valid_coord), "protected rows need coordinates")
    keep = valid_id & valid_coord & ~owner & ~blocked & ~old
    return keep, dict(
        rows=len(valid_id),
        invalid_id=int((~valid_id).sum()),
        invalid_coordinates=int((~valid_coord).sum()),
        blocked_owner=int(owner.sum()),
        blocked_id=int(blocked.sum()),
        old_id=int(old.sum()),
        eligible_before_radius=int(keep.sum()),
    )


def as_records(a: Columns, mask: NDArray[np.bool_], seed: int) -> NDArray[np.void]:
    rows = np.empty(int(mask.sum()), dtype=RECORD)
    for k in COLUMNS:
        rows[k] = a[k][mask]
    rows["sample_hash"] = splitmix64(rows["object_id"], seed)
    return rows


def mark_neighbors(
    tree: cKDTree,
    positions: NDArray[np.float64],
    candidates: NDArray[np.void],
    reasons: NDArray[np.uint16],
    radius_arcsec: float,
    bit: int,
    coordinates: tuple[NDArray[np.float64], NDArray[np.float64]],
    ids: NDArray[np.int64] | None = None,
) -> None:
    radius = radius_arcsec * ARCSEC_RAD
    threshold = radius + ANGLE_BAND_RAD
    chord = 2 * np.sin((radius + ANGLE_BAND_RAD) / 2) + QUERY_PADDING_CHORD
    for start in range(0, len(positions), 256):
        points = positions[start : start + 256]
        lists = tree.query_ball_point(points, chord, workers=1)
        for j, neighbors in enumerate(lists):
            if not neighbors:
                continue
            ix = np.asarray(neighbors, dtype=np.int64)
            if ids is not None:
                ix = ix[candidates["object_id"][ix] != ids[start + j]]
            if not len(ix):
                continue
            separation = angle(tree.data[ix], points[j])
            near = separation <= threshold
            for k in np.flatnonzero(np.abs(separation - threshold) <= 1e-12):
                q = candidates[ix[k]]
                near[k] = boundary_within(
                    q["ra_deg"],
                    q["dec_deg"],
                    coordinates[0][start + j],
                    coordinates[1][start + j],
                    threshold,
                )
            reasons[ix[near]] |= np.uint16(bit)


def select_objects(cfg: SelectionConfig) -> NDArray[np.void]:
    manifest = json.loads(cfg.registry_manifest.read_text())
    plans = registry_plan(manifest, cfg.registry_manifest.parent)
    g = load_guards(cfg.guards)
    reservoirs = {f: Reservoir(cfg.reservoir_per_field, cfg.merge_rows) for f in (1, 2)}
    counts: dict[int, Counter[str]] = {f: Counter() for f in (1, 2)}
    seen: list[NDArray[np.int64]] = []
    for a in iter_registry(plans, cfg.batch_rows):
        seen.append(a["object_id"][a["object_id"] != NULL_ID])
        mask, c = metadata_keep(a, g)
        f = int(a["field_code"][0])
        counts[f].update(c)
        reservoirs[f].push(as_records(a, mask, cfg.sample_seed))
    all_ids = np.concatenate(seen)
    all_ids.sort(kind="quicksort")
    check(not np.any(all_ids[1:] == all_ids[:-1]), "duplicate object_id in the F/S registry")
    del all_ids, seen
    logger.info("registry pass 1: %s", {f: dict(c) for f, c in counts.items()})
    candidates = np.concatenate([reservoirs[f].finish() for f in (1, 2)])
    del reservoirs
    check(len(candidates) > 0, "empty candidate reservoir")
    reasons = np.zeros(len(candidates), dtype=np.uint16)
    tree = cKDTree(
        xyz(candidates["ra_deg"], candidates["dec_deg"]),
        compact_nodes=True,
        balanced_tree=True,
        copy_data=False,
    )
    mark_neighbors(
        tree,
        xyz(g["guard_ra_deg"], g["guard_dec_deg"]),
        candidates,
        reasons,
        cfg.guard_radius_arcsec,
        GUARD_BIT,
        (g["guard_ra_deg"], g["guard_dec_deg"]),
    )
    for a in iter_registry(plans, cfg.batch_rows):
        finite = finite_coords(a["ra_deg"], a["dec_deg"])
        protected = contains(g["blocked_tiles"], a["tile"]) | contains(
            g["blocked_ids"], a["object_id"]
        )
        check(not np.any(protected & ~finite), "protected rows need coordinates")
        for rows, bit, radius, own in (
            (finite & (a["object_id"] != NULL_ID), NEIGHBOR_BIT, cfg.neighbor_radius_arcsec, True),
            (finite & (a["object_id"] == NULL_ID), UNKNOWN_BIT, cfg.neighbor_radius_arcsec, False),
            (finite & protected, GUARD_BIT, cfg.guard_radius_arcsec, False),
        ):
            ra, dec = a["ra_deg"][rows], a["dec_deg"][rows]
            mark_neighbors(
                tree,
                xyz(ra, dec),
                candidates,
                reasons,
                radius,
                bit,
                (ra, dec),
                ids=a["object_id"][rows] if own else None,
            )
    selected = np.zeros(len(candidates), dtype=bool)
    for f in (1, 2):
        eligible = np.flatnonzero((candidates["field_code"] == f) & ((reasons & FATAL) == 0))
        selected[eligible[: cfg.target_per_field]] = True
        logger.info(
            "field %d: reservoir %d, survived radius %d, selected %d, shortage %d",
            f,
            int(np.sum(candidates["field_code"] == f)),
            len(eligible),
            int(selected[candidates["field_code"] == f].sum()),
            max(0, cfg.target_per_field - len(eligible)),
        )
    return candidates[selected]


def write_selected(path: Path, rows: NDArray[np.void]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(path, SCHEMA, compression="zstd", use_dictionary=False) as writer:
        for start in range(0, len(rows), WRITE_ROWS):
            part = rows[start : start + WRITE_ROWS]
            values = {k: pa.array(part[k], type=SCHEMA.field(k).type) for k in RECORD_COLUMNS}
            writer.write_table(pa.table(values, schema=SCHEMA))


def read_header(stream: BinaryIO) -> fits.Header:
    blocks: list[bytes] = []
    while True:
        block = stream.read(FITS_BLOCK)
        check(len(block) == FITS_BLOCK and block.isascii(), "complete ASCII FITS header")
        blocks.append(block)
        if any(block[i : i + 8] == b"END     " for i in range(0, FITS_BLOCK, 80)):
            return fits.Header.fromstring(b"".join(blocks).decode("ascii"), sep="")


def catalog_layout(path: Path) -> tuple[int, int, int, dict[str, int]]:
    with path.open("rb") as stream:
        primary = read_header(stream)
        check(primary.get("SIMPLE") is True and primary.get("NAXIS") == 0, "empty primary HDU")
        header = read_header(stream)
        data_offset = stream.tell()
    check(
        header.get("XTENSION") == "BINTABLE"
        and header.get("NAXIS") == 2
        and header.get("PCOUNT", 0) == 0
        and header.get("GCOUNT", 1) == 1,
        "plain binary table without heap",
    )
    width, rows = int(header["NAXIS1"]), int(header["NAXIS2"])
    layout: dict[str, int] = {}
    position = 0
    for i in range(1, int(header["TFIELDS"]) + 1):
        name, form = header[f"TTYPE{i}"], str(header[f"TFORM{i}"])
        match = re.fullmatch(r"([0-9]*)([LXBIJKAEDCM])", form)
        if match is None:
            raise ValueError(f"unsupported TFORM {form} in {path}")
        repeat, code = int(match[1] or "1"), match[2]
        if name in CATALOG_FORMS:
            check(
                repeat == 1
                and code == CATALOG_FORMS[name]
                and header.get(f"TSCAL{i}", 1) == 1
                and header.get(f"TZERO{i}", 0) == 0,
                f"unscaled scalar column {name}",
            )
            layout[name] = position
        position += (repeat + 7) // 8 if code == "X" else repeat * TFORM_SIZES[code]
    check(position == width and set(layout) == set(CATALOG_FORMS), f"catalog layout {path}")
    return data_offset, width, rows, layout


def column_words(block: NDArray[np.uint8], offset: int, size: int) -> NDArray[Any]:
    raw = np.ascontiguousarray(block[:, offset : offset + size])
    return raw.view(np.dtype(f">u{size}"))[:, 0].astype(np.dtype(f"u{size}"))


def read_catalog_rows(
    path: Path, rows: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32]]:
    data_offset, width, nrows, layout = catalog_layout(path)
    check(len(rows) > 0 and rows.min() >= 0 and rows.max() < nrows, f"row range {path}")
    table = np.memmap(path, dtype=np.uint8, mode="r", offset=data_offset, shape=(nrows, width))
    block = np.asarray(table[rows])
    del table
    ids = column_words(block, layout["OBJECT_ID"], 8).view(np.int64)
    flux = np.empty((len(rows), len(BANDS)), np.float32)
    error = np.empty((len(rows), len(BANDS)), np.float32)
    for j in range(len(BANDS)):
        flux.view(np.uint32)[:, j] = column_words(block, layout[FLUX[j]], 4)
        error.view(np.uint32)[:, j] = column_words(block, layout[ERROR[j]], 4)
    return ids, flux, error


def source_photometry(
    source: dict[str, Any],
    catalog_dir: Path,
    selection_rows: NDArray[np.int64],
    object_id: NDArray[np.int64],
    source_row: NDArray[np.int64],
) -> dict[str, NDArray[Any]]:
    check(source["field"] in FIELD_CODES, f"F/S catalog source {source['source_file_index']}")
    ix = selection_rows[np.argsort(source_row[selection_rows], kind="stable")]
    path = catalog_dir / Path(source["path"]).name
    found, flux, error = read_catalog_rows(path, source_row[ix].astype(np.int64))
    check(np.array_equal(found, object_id[ix]), f"catalog object ids {path}")
    return dict(
        object_id=object_id[ix],
        selection_row=ix.astype(np.int64),
        flux_ujy=flux,
        fluxerr_ujy=error,
        valid_measurement=np.isfinite(flux) & np.isfinite(error) & (error > 0),
        bands=np.asarray(BANDS, dtype="U16"),
    )


def project_photometry(cfg: SelectionConfig) -> None:
    sources = json.loads(cfg.registry_manifest.read_text())["sources"]
    table = pq.read_table(cfg.selected, columns=["object_id", "source_file_index", "source_row"])
    object_id = table["object_id"].to_numpy()
    index = table["source_file_index"].to_numpy()
    source_row = table["source_row"].to_numpy()
    cfg.raw14_dir.mkdir(parents=True, exist_ok=True)
    for si in np.unique(index).tolist():
        check(sources[si]["source_file_index"] == si, f"catalog source index {si}")
        arrays: dict[str, Any] = source_photometry(
            sources[si], cfg.mer_catalog_dir, np.flatnonzero(index == si), object_id, source_row
        )
        np.savez(cfg.raw14_dir / f"raw14_{si:04d}.npz", **arrays)
        logger.info(
            "raw14 source %d: %d rows, at least two valid bands %d",
            si,
            len(arrays["object_id"]),
            int((arrays["valid_measurement"].sum(1) >= 2).sum()),
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


@cli.command("select")
@CONFIG
def select_command(config_path: Path) -> None:
    cfg = load_yaml_as(str(config_path), SelectionConfig)
    rows = select_objects(cfg)
    write_selected(cfg.selected, rows)
    logger.info("selected %d objects -> %s", len(rows), cfg.selected)


@cli.command("photometry")
@CONFIG
def photometry_command(config_path: Path) -> None:
    project_photometry(load_yaml_as(str(config_path), SelectionConfig))


if __name__ == "__main__":
    cli()
