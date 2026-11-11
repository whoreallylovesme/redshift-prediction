from __future__ import annotations

from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import click
import h5py
import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.typing import NDArray
from pydantic import BaseModel

from lib.utils.log import get_logger
from lib.utils.utils import load_yaml_as

logger = get_logger()

COLUMNS = ("PHZ_MEDIAN", "PHZ_MODE_1", "PHZ_70_INT", "PHZ_CLASSIFICATION")
_WANTED: dict[str, NDArray[np.int64]] = {}


class IdSource(BaseModel):
    path: Path
    column: str = "object_id"
    indices: list[str] = []


class Lookup(BaseModel):
    ids: list[IdSource]
    out: Path


class PhzLookupConfig(BaseModel):
    phz_dir: Path
    pattern: str = "*.fits"
    workers: int = 8
    lookups: list[Lookup]


def read_ids(source: IdSource) -> NDArray[np.int64]:
    path, column = source.path, source.column
    if path.suffix == ".h5":
        with h5py.File(path, "r") as h:
            values = h[column][:]
            if source.indices:
                values = values[np.concatenate([h[k][:] for k in source.indices])]
    elif path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as packed:
            values = packed[column]
    else:
        values = pd.read_parquet(path, columns=[column])[column].to_numpy()
    out: NDArray[np.int64] = np.asarray(values).astype(np.int64)
    return out


def load_ids(sources: Sequence[IdSource]) -> NDArray[np.int64]:
    out: NDArray[np.int64] = np.unique(np.concatenate([read_ids(s) for s in sources]))
    return out


def init(ids: NDArray[np.int64]) -> None:
    _WANTED["ids"] = ids


def scan(path: str) -> dict[str, NDArray[Any]] | None:
    with fits.open(path, memmap=True) as hdus:
        data = hdus[1].data
        oid = np.asarray(data["OBJECT_ID"]).astype(np.int64)
        keep = np.isin(oid, _WANTED["ids"])
        if not keep.any():
            return None
        out: dict[str, NDArray[Any]] = {"object_id": oid[keep]}
        for column in COLUMNS:
            value = np.asarray(data[column])[keep]
            out[column] = value.reshape(len(value), -1) if value.ndim > 1 else value
        return out


def lookup(ids: NDArray[np.int64], files: list[str], workers: int) -> dict[str, NDArray[Any]]:
    with Pool(workers, initializer=init, initargs=(ids,)) as pool:
        parts = [p for p in pool.imap(scan, files) if p is not None]
    if not parts:
        raise ValueError("no requested id found in the PHZ catalogue")
    result = {k: np.concatenate([p[k] for p in parts]) for k in ("object_id", *COLUMNS)}
    if len(np.unique(result["object_id"])) != len(result["object_id"]):
        raise ValueError("duplicate ids across PHZ files")
    return result


@click.command()
@click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True
)
def main(config_path: Path) -> None:
    cfg = load_yaml_as(str(config_path), PhzLookupConfig)
    files = sorted(str(p) for p in cfg.phz_dir.glob(cfg.pattern))
    for job in cfg.lookups:
        ids = load_ids(job.ids)
        result = lookup(ids, files, cfg.workers)
        job.out.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = dict(result)
        np.savez(job.out, **arrays)
        logger.info(
            "requested %d found %d in %d files -> %s",
            len(ids), len(result["object_id"]), len(files), job.out,
        )


if __name__ == "__main__":
    main()
