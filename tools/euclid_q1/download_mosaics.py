from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click
import numpy as np
import pyarrow.parquet as pq
from astropy.io.votable import parse_single_table
from pydantic import BaseModel

from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as

logger = get_logger()

IMAGE_BAND_NAMES = ("VIS", "Y", "J", "H")
KINDS = ("science", "rms", "flag")
CHUNK_BYTES = 1 << 20

type SourceMap = dict[str, dict[str, dict[str, str]]]


class DownloadMosaicsConfig(BaseModel):
    selected: Path
    source_map: Path
    mosaic_dirs: list[str]
    sia_url: str = "https://irsa.ipac.caltech.edu/SIA"
    sia_collection: str = "euclid_DpdMerBksMosaic"
    sia_radius_deg: float = 0.0003
    sia_max_records: int = 200
    archive_prefix: str = "https://irsa.ipac.caltech.edu/ibe/data/euclid/q1/"
    mirror_prefix: str = "https://nasa-irsa-euclid-q1.s3.us-east-1.amazonaws.com/q1/"
    download_workers: int = 8
    timeout_s: float = 600.0


def owner_key(field_code: int, tile: int) -> str:
    return f"{'F' if field_code == 1 else 'S'}{tile}"


def selected_owners(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path, columns=["object_id", "field_code", "tile", "ra_deg", "dec_deg"])
    t = {k: table[k].to_numpy() for k in table.column_names}
    owners: dict[int, dict[str, Any]] = {}
    for tile in np.unique(t["tile"]).tolist():
        rows = np.flatnonzero(t["tile"] == tile)
        r = rows[np.argmin(t["object_id"][rows])]
        owners[tile] = dict(
            tile=tile,
            field_code=int(t["field_code"][r]),
            object_id=int(t["object_id"][r]),
            ra_deg=float(t["ra_deg"][r]),
            dec_deg=float(t["dec_deg"][r]),
        )
    return [owners[k] for k in sorted(owners)]


def sia_request(cfg: DownloadMosaicsConfig, owner: dict[str, Any]) -> str:
    params = urllib.parse.urlencode(
        dict(
            POS=f"CIRCLE {owner['ra_deg']:.12f} {owner['dec_deg']:.12f} {cfg.sia_radius_deg}",
            COLLECTION=cfg.sia_collection,
            FACILITY="Euclid",
            MAXREC=cfg.sia_max_records,
            RESPONSEFORMAT="VOTABLE",
        )
    )
    return f"{cfg.sia_url}?{params}"


def parse_owner_sources(raw: bytes, tile: int) -> dict[str, dict[str, str]]:
    tree = ET.fromstring(raw)
    info = [
        x
        for x in tree.iter()
        if x.tag.rsplit("}", 1)[-1] == "INFO" and x.get("name") == "QUERY_STATUS"
    ]
    check(info and all(x.get("value") == "OK" for x in info), "complete SIA query")
    table = parse_single_table(io.BytesIO(raw)).to_table(use_names_over_ids=True)
    check(
        {"energy_bandpassname", "access_url"} <= set(table.colnames), "SIA response columns"
    )
    found: dict[str, dict[str, set[str]]] = {
        b: {k: set() for k in KINDS} for b in IMAGE_BAND_NAMES
    }
    for row in table:
        band, url = str(row["energy_bandpassname"]), str(row["access_url"])
        if band not in IMAGE_BAND_NAMES or f"/MER/{tile}/" not in url:
            continue
        name = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
        kind = (
            "science"
            if name.startswith("EUC_MER_BGSUB-MOSAIC-")
            else "rms"
            if "-RMS_" in name
            else "flag"
            if "-FLAG_" in name
            else None
        )
        if kind:
            found[band][kind].add(url)
    sources: dict[str, dict[str, str]] = {}
    for band in IMAGE_BAND_NAMES:
        sources[band] = {}
        for kind in KINDS:
            urls = sorted(found[band][kind])
            check(len(urls) == 1, f"one {band}/{kind} mosaic for tile {tile}, got {len(urls)}")
            sources[band][kind] = urls[0]
    return sources


def discover_sources(cfg: DownloadMosaicsConfig) -> None:
    owners: dict[str, Any] = {}
    if cfg.source_map.exists():
        owners = json.loads(cfg.source_map.read_text())["owners"]
    for owner in selected_owners(cfg.selected):
        if str(owner["tile"]) in owners:
            continue
        with urllib.request.urlopen(sia_request(cfg, owner), timeout=cfg.timeout_s) as response:
            raw = response.read()
        owners[str(owner["tile"])] = dict(
            field_code=owner["field_code"],
            image_sources=parse_owner_sources(raw, owner["tile"]),
        )
        logger.info("discovered mosaics for %s", owner_key(owner["field_code"], owner["tile"]))
    cfg.source_map.parent.mkdir(parents=True, exist_ok=True)
    cfg.source_map.write_text(json.dumps(dict(owners=owners), indent=1) + "\n")


def transport_url(cfg: DownloadMosaicsConfig, url: str, tile: int) -> str:
    check(url.startswith(f"{cfg.archive_prefix}MER/{tile}/"), f"archive mosaic url {url}")
    if not cfg.mirror_prefix:
        return url
    return cfg.mirror_prefix + url[len(cfg.archive_prefix) :]


def existing(cfg: DownloadMosaicsConfig, key: str, tile: int, name: str) -> bool:
    return any((Path(t.format(tile=tile, key=key)) / name).is_file() for t in cfg.mosaic_dirs)


def download(url: str, target: Path, timeout: float) -> int:
    partial = target.with_name(target.name + ".partial")
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    written = 0
    with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as f:
        expected = int(response.headers["Content-Length"])
        while chunk := response.read(CHUNK_BYTES):
            f.write(chunk)
            written += len(chunk)
        f.flush()
        os.fsync(f.fileno())
    check(written == expected, f"truncated download {url}: {written} of {expected} bytes")
    os.replace(partial, target)
    return written


def fetch_mosaics(cfg: DownloadMosaicsConfig) -> None:
    sources: SourceMap = {
        k: v["image_sources"] for k, v in json.loads(cfg.source_map.read_text())["owners"].items()
    }
    jobs: list[tuple[str, Path]] = []
    for owner in selected_owners(cfg.selected):
        tile = owner["tile"]
        key = owner_key(owner["field_code"], tile)
        target_dir = Path(cfg.mosaic_dirs[0].format(tile=tile, key=key))
        for band in IMAGE_BAND_NAMES:
            for kind in KINDS:
                name = f"{tile}_{band}_{kind}.fits"
                if existing(cfg, key, tile, name):
                    continue
                url = transport_url(cfg, sources[str(tile)][band][kind], tile)
                target_dir.mkdir(parents=True, exist_ok=True)
                jobs.append((url, target_dir / name))
    logger.info("downloading %d mosaic files", len(jobs))
    failed = []
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as pool:
        futures = {pool.submit(download, url, path, cfg.timeout_s): path for url, path in jobs}
        for future in as_completed(futures):
            path = futures[future]
            try:
                logger.info("%s: %d bytes", path, future.result())
            except Exception as exc:
                failed.append(path)
                logger.error("%s: %s", path, exc)
    if failed:
        raise click.ClickException(f"{len(failed)} mosaic downloads failed; rerun to resume")


CONFIG = click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)


@click.group()
def cli() -> None:
    pass


@cli.command("discover")
@CONFIG
def discover_command(config_path: Path) -> None:
    discover_sources(load_yaml_as(str(config_path), DownloadMosaicsConfig))


@cli.command("fetch")
@CONFIG
def fetch_command(config_path: Path) -> None:
    fetch_mosaics(load_yaml_as(str(config_path), DownloadMosaicsConfig))


if __name__ == "__main__":
    cli()
