from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import click
import h5py
import numpy as np
from astropy.io import fits
from numpy.typing import NDArray
from pydantic import BaseModel

from lib.dataloaders.euclid_transforms import BANDS
from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as

logger = get_logger()

FLUX_COLUMNS = tuple(f"FLUX_{band}_2FWHM_APER" for band in BANDS)
ERROR_COLUMNS = tuple(f"FLUXERR_{band}_2FWHM_APER" for band in BANDS)
ANCILLARY_COLUMNS = (
    "GAL_EBV",
    "GAL_EBV_ERR",
    "FLUX_DETECTION_TOTAL",
    "FLUXERR_DETECTION_TOTAL",
    "FLUX_NIR_STACK_2FWHM_APER",
    "FLUXERR_NIR_STACK_2FWHM_APER",
)
TILE_PATTERN = re.compile(r"TILE(\d+)")
OUTPUT_SUFFIX = "_native_photometry.npz"

type Arrays = dict[str, NDArray[Any]]
type TileTask = tuple[Path, int, int, dict[str, NDArray[np.int64]]]


class MerSet(BaseModel):
    name: str
    source: Path
    indices: str | None = None
    ids: Path | None = None
    split: str = "test"


class ExtractConfig(BaseModel):
    mer_dir: Path
    output_dir: Path
    sets: list[MerSet]
    mer_glob: str = "EUC_MER_FINAL-CAT_TILE*.fits"
    workers: int = 8


def measurement_states(flux: NDArray[Any], error: NDArray[Any]) -> NDArray[np.uint8]:
    check(flux.shape == error.shape, "flux/error shapes differ")
    state = (
        np.isfinite(flux).astype(np.uint8)
        | (np.isfinite(error).astype(np.uint8) << 1)
        | ((np.isfinite(error) & (error > 0)).astype(np.uint8) << 2)
        | ((np.isfinite(flux) & (flux < 0)).astype(np.uint8) << 3)
    )
    return np.asarray(state, dtype=np.uint8)


def request(spec: MerSet) -> Arrays:
    if spec.source.suffix == ".h5":
        with h5py.File(spec.source, "r") as h:
            oid, tile, field = h["object_id"][:], h["owner_tile"][:], h["field_code"][:]
            if spec.indices is not None:
                rows = h[spec.indices][:]
                oid, tile, field = oid[rows], tile[rows], field[rows]
    else:
        with np.load(spec.source, allow_pickle=False) as z:
            oid, tile, field = z["object_id"], z["tile"], z["field_code"]
    if spec.ids is not None:
        want = np.load(spec.ids, allow_pickle=False)
        pos = {int(v): i for i, v in enumerate(oid)}
        rows = np.array([pos[int(v)] for v in want], np.int64)
        oid, tile, field = oid[rows], tile[rows], field[rows]
    check(len(np.unique(oid)) == len(oid), f"{spec.name}: duplicate ids")
    return dict(
        object_id=oid.astype(np.int64),
        tile=tile.astype(np.int64),
        field_code=field.astype(np.uint8),
    )


def tile_id(name: str) -> int:
    match = TILE_PATTERN.search(name)
    if match is None:
        raise ValueError(f"no tile id in {name}")
    return int(match.group(1))


def mer_files(mer_dir: Path, pattern: str) -> dict[int, tuple[int, Path]]:
    files: dict[int, tuple[int, Path]] = {}
    for index, path in enumerate(sorted(mer_dir.glob(pattern))):
        tile = tile_id(path.name)
        check(tile not in files, f"tile {tile} has several MER catalogues")
        files[tile] = (index, path)
    return files


def column(data: Any, name: str, rows: NDArray[np.int64]) -> NDArray[Any]:
    values: NDArray[Any] = np.asarray(data[name])[rows]
    return values


def extract_tile(task: TileTask) -> dict[str, Arrays]:
    path, index, tile, requests = task
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdus:
        table = hdus[1]
        check(isinstance(table, fits.BinTableHDU), f"{path.name}: extension 1 not a table")
        check(int(table.header["TILEID"]) == tile, f"{path.name}: TILEID differs from {tile}")
        data = table.data
        vis_null = table.columns["VIS_DET"].null
        file_ids = np.asarray(data["OBJECT_ID"])
        order = np.argsort(file_ids, kind="stable")
        sorted_ids = file_ids[order]
        results: dict[str, Arrays] = {}
        for name, want in requests.items():
            left = np.searchsorted(sorted_ids, want, side="left")
            right = np.searchsorted(sorted_ids, want, side="right")
            check(((right - left) == 1).all(), f"{name} tile {tile}: ids not found exactly once")
            rows = order[left]
            vis = column(data, "VIS_DET", rows)
            results[name] = dict(
                object_id=want.astype(np.int64),
                source_row=rows.astype(np.int64),
                ra_deg=column(data, "RIGHT_ASCENSION", rows).astype(np.float64),
                dec_deg=column(data, "DECLINATION", rows).astype(np.float64),
                flux=np.column_stack([column(data, c, rows) for c in FLUX_COLUMNS]).astype(
                    np.float64
                ),
                error=np.column_stack([column(data, c, rows) for c in ERROR_COLUMNS]).astype(
                    np.float64
                ),
                ancillary_values=np.column_stack(
                    [column(data, c, rows) for c in ANCILLARY_COLUMNS]
                ).astype(np.float64),
                vis_det=vis.astype(np.int64),
                vis_det_missing=(
                    np.zeros(len(rows), dtype=bool) if vis_null is None else vis == vis_null
                ),
                tile=np.full(len(rows), tile, dtype=np.int64),
                source_file_index=np.full(len(rows), index, dtype=np.int32),
            )
    return results


def assemble(spec: MerSet, req: Arrays, pieces: list[Arrays]) -> Arrays:
    merged = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    pos = {int(v): i for i, v in enumerate(merged["object_id"])}
    check(
        len(pos) == len(merged["object_id"]) and set(pos) == set(req["object_id"].tolist()),
        f"{spec.name}: assembled ids differ from request",
    )
    idx = np.array([pos[int(v)] for v in req["object_id"]], dtype=np.int64)
    arrays = {k: v[idx] for k, v in merged.items()}
    check(np.array_equal(arrays["tile"], req["tile"]), f"{spec.name}: owner tile mismatch")
    arrays["field_code"] = req["field_code"].astype(np.uint8)
    arrays["split"] = np.full(len(idx), spec.split, dtype="<U5")
    arrays["measurement_state"] = measurement_states(arrays["flux"], arrays["error"])
    arrays["valid"] = (arrays["measurement_state"] & 7) == 7
    arrays["ancillary_finite"] = np.isfinite(arrays["ancillary_values"])
    arrays["bands"] = np.asarray(BANDS)
    arrays["ancillary_columns"] = np.asarray(ANCILLARY_COLUMNS)
    return arrays


def extract(config: ExtractConfig) -> dict[str, Arrays]:
    files = mer_files(config.mer_dir, config.mer_glob)
    requests = {spec.name: request(spec) for spec in config.sets}
    tasks: dict[int, dict[str, NDArray[np.int64]]] = {}
    for name, r in requests.items():
        for tile in np.unique(r["tile"]).tolist():
            check(int(tile) in files, f"{name}: no MER catalogue for tile {tile}")
            tasks.setdefault(int(tile), {})[name] = r["object_id"][r["tile"] == tile]
    order = sorted(tasks, key=lambda t: -files[t][1].stat().st_size)
    logger.info(
        "MER extraction: %d tiles, sets %s",
        len(order),
        {k: len(v["object_id"]) for k, v in requests.items()},
    )
    pieces: dict[str, list[Arrays]] = {name: [] for name in requests}
    work = [(files[t][1], files[t][0], t, tasks[t]) for t in order]
    with ProcessPoolExecutor(max_workers=config.workers) as pool:
        for done, results in enumerate(pool.map(extract_tile, work), start=1):
            for name, arrays in results.items():
                pieces[name].append(arrays)
            if done % 20 == 0:
                logger.info("MER tiles %d/%d", done, len(work))
    return {
        spec.name: assemble(spec, requests[spec.name], pieces[spec.name]) for spec in config.sets
    }


@click.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def main(config_path: Path) -> None:
    config = load_yaml_as(str(config_path), ExtractConfig)
    out = extract(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for name, arrays in out.items():
        path = config.output_dir / f"{name}{OUTPUT_SUFFIX}"
        np.savez_compressed(path, **dict[str, Any](arrays))
        logger.info(
            "%s: %d rows, %d tiles -> %s",
            name,
            len(arrays["object_id"]),
            len(np.unique(arrays["tile"])),
            path,
        )


if __name__ == "__main__":
    main()
