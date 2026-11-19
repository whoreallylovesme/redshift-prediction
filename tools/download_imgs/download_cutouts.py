#!/usr/bin/env python3
"""Download Legacy Survey DR10 cutouts via the cutout API.

Each request returns a single FITS with griz bands (4 channels).
Rate-limited to ~2 concurrent requests to avoid 429s.

Usage:
  python download_cutouts.py --dataset teddy --split C
  python download_cutouts.py --dataset teddy --split A,B,C,D --concurrency 2
"""
from __future__ import annotations

import asyncio
import json
import sys
from functools import partial
from io import BytesIO
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
import click
import numpy as np
from astropy.io import fits
from tqdm import tqdm

if TYPE_CHECKING or __package__:
    from tools.download_imgs import survey
    from tools.download_imgs.workspace import download_destination
else:
    import survey
    from workspace import download_destination

# ── Config ────────────────────────────────────────────────────────────────────
STAMP_SIZE = 224
PIXSCALE = 0.262  # arcsec/pixel
BANDS = ["g", "r", "i", "z"]
CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.fits"

# ── SDSS ──────────────────────────────────────────────────────────────────────
SDSS_DR = survey.SDSS_DR
SDSS_BATCH = survey.SDSS_BATCH
SDSS_TIMEOUT = survey.SDSS_TIMEOUT
SDSS_ENDPOINT = survey.SDSS_ENDPOINT

MAX_RETRIES = 8
RETRY_BACKOFF = 2.0

_OUT = click.Path(file_okay=False, dir_okay=True, path_type=Path)


# The cutout CLI historically uses a longer retry policy for SDSS.
_try_sdss = survey._try_sdss
_sdss_batch = partial(survey._sdss_batch, max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF)
fetch_coordinates = partial(
    survey.fetch_coordinates, max_retries=MAX_RETRIES, backoff=RETRY_BACKOFF,
)


# ── Download one cutout ──────────────────────────────────────────────────────

async def download_one_cutout(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    objid: int,
    ra: float,
    dec: float,
    dest_path: Path,
) -> bool:
    """Download a single cutout FITS, extract griz, save as .npz. Returns True on success."""
    if dest_path.exists():
        return True

    url = (
        f"{CUTOUT_URL}?ra={ra}&dec={dec}"
        f"&size={STAMP_SIZE}&layer=ls-dr10&pixscale={PIXSCALE}&bands=griz"
    )

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url) as resp:
                    if resp.status == 429:
                        wait = RETRY_BACKOFF * (2 ** attempt)
                        await asyncio.sleep(wait)
                        continue
                    if resp.status == 404:
                        return False
                    resp.raise_for_status()
                    data = await resp.read()

                    with fits.open(BytesIO(data), memmap=False) as hdul:
                        cube = np.array(hdul[0].data, dtype=np.float32)  # (4, H, W)

                    if cube.ndim != 3 or cube.shape[0] != 4:
                        return False

                    # (4, H, W) → (H, W, 4)
                    image = np.moveaxis(cube, 0, -1)
                    if image.shape != (STAMP_SIZE, STAMP_SIZE, 4):
                        # Edge cutouts may be smaller — pad to full size
                        padded = np.zeros((STAMP_SIZE, STAMP_SIZE, 4), dtype=np.float32)
                        h, w = min(image.shape[0], STAMP_SIZE), min(image.shape[1], STAMP_SIZE)
                        padded[:h, :w, :] = image[:h, :w, :]
                        image = padded

                    with download_destination(dest_path) as stream:
                        np.savez_compressed(stream, image=image)
                    return True

            except (aiohttp.ClientError, TimeoutError) as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"    FAIL {objid} — {e}", flush=True)
                    return False
                await asyncio.sleep(RETRY_BACKOFF * (2 ** min(attempt, 4)))

    return False


# ── Pipeline ─────────────────────────────────────────────────────────────────

async def download_pipeline(
    objects: list[tuple[int, float, float, Path]],
    concurrency: int,
    timeout: float,
) -> tuple[int, int, list[int]]:
    """Download all cutouts. Returns (saved, failed, failed_ids)."""
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 2, ttl_dns_cache=300)
    timeout_cfg = aiohttp.ClientTimeout(total=timeout, connect=30)

    saved = 0
    failed_ids: list[int] = []
    async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as session:
        with tqdm(total=len(objects), unit="object", file=sys.stdout) as progress:
            for chunk in batched(objects, 50, strict=False):
                results = await asyncio.gather(*(
                    download_one_cutout(session, semaphore, objid, ra, dec, dest)
                    for objid, ra, dec, dest in chunk
                ))
                for (objid, _, _, _), ok in zip(chunk, results, strict=True):
                    if ok:
                        saved += 1
                    else:
                        failed_ids.append(objid)
                progress.set_postfix(saved=saved, failed=len(failed_ids), refresh=False)
                progress.update(len(chunk))
    return saved, len(failed_ids), failed_ids


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option(
    "--teddy-json",
    default="teddy_ids.json",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--happy-json",
    default="happy_ids.json",
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--outdir", default="legacy_out", type=_OUT)
@click.option("--dataset", default="teddy,happy", show_default=True)
@click.option("--split", default="A,B,C,D", show_default=True)
@click.option(
    "--concurrency",
    default=2,
    show_default=True,
    help="Parallel requests (keep low, API rate-limits)",
)
@click.option("--max-per-split", default=None, type=int, help="Limit per split (testing)")
@click.option("--timeout", default=30.0, show_default=True)
@click.option("--dry-run", is_flag=True)
def main(
    teddy_json: Path,
    happy_json: Path,
    outdir: Path,
    dataset: str,
    split: str,
    concurrency: int,
    max_per_split: int | None,
    timeout: float,
    dry_run: bool,
) -> None:
    selected_datasets = [d.strip().lower() for d in dataset.split(",")]
    selected_splits = [s.strip().upper() for s in split.split(",")]
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets: {selected_datasets} | Splits: {selected_splits}")
    print(f"Concurrency: {concurrency} | Output: {outdir.resolve()}")

    # ── Load IDs ──────────────────────────────────────────────────────────────
    objid_to_dest: dict[int, Path] = {}
    for ds in selected_datasets:
        json_path = teddy_json if ds == "teddy" else happy_json
        raw = json.loads(json_path.read_text())
        for s in selected_splits:
            if s in raw:
                ids = [int(x) for x in raw[s]]
                if max_per_split:
                    ids = ids[:max_per_split]
                split_dir = outdir / ds / s
                split_dir.mkdir(parents=True, exist_ok=True)
                new = 0
                for oid in ids:
                    if not (split_dir / f"{oid}.npz").exists():
                        objid_to_dest[oid] = split_dir
                        new += 1
                print(f"  {ds}/{s}: {len(ids)} total, {new} to download", flush=True)

    if not objid_to_dest:
        print("\nAll objects already downloaded!")
        return

    print(f"\nTotal objects to download: {len(objid_to_dest)}")

    # ── Step 1: Coordinates ───────────────────────────────────────────────────
    print("\n[1/2] Fetching coordinates from SDSS...")
    cache_path = outdir / "coords.parquet"
    coords = fetch_coordinates(list(objid_to_dest.keys()), cache_path)
    if coords.empty:
        print("No coordinates found!")
        return
    coords = coords[coords["objid"].isin(objid_to_dest)]
    found_ids = set(coords["objid"].tolist())
    no_coords_ids = [oid for oid in objid_to_dest if oid not in found_ids]
    print(f"  Got {len(coords)}/{len(objid_to_dest)} coordinates")
    if no_coords_ids:
        print(f"  WARNING: {len(no_coords_ids)} objects not found in SDSS", flush=True)

    if dry_run:
        rate_est = 0.7 * concurrency  # ~0.7 obj/s per concurrent slot (2.5s response + overhead)
        est_hours = len(coords) / rate_est / 3600
        npz_gb = len(coords) * STAMP_SIZE * STAMP_SIZE * 4 * 4 * 0.5 / (1024**3)
        print(f"\n{'=' * 50}")
        print("  DRY RUN SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Objects:       {len(coords):,}")
        print(f"  No coords:     {len(no_coords_ids):,}")
        print("  Bands:         griz (4 channels)")
        print(f"  Output .npz:   ~{npz_gb:.1f} GB")
        print(f"  Est time:      ~{est_hours:.1f}h (concurrency={concurrency})")
        print(f"{'=' * 50}")
        return

    # ── Step 2: Download cutouts ──────────────────────────────────────────────
    print(f"\n[2/2] Downloading cutouts (concurrency={concurrency})...")
    objects = [
        (
            int(row.objid),
            float(row.ra),
            float(row.dec),
            objid_to_dest[int(row.objid)] / f"{int(row.objid)}.npz",
        )
        for row in coords.itertuples(index=False)
    ]

    total_saved, total_failed, failed_ids = asyncio.run(
        download_pipeline(objects, concurrency, timeout)
    )

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "total_objects": len(objid_to_dest),
        "saved": total_saved,
        "failed_no_coords": no_coords_ids,
        "failed_cutout": failed_ids,
        "failed_total": total_failed + len(no_coords_ids),
        "bands": BANDS,
    }
    report_path = outdir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\nDone! saved={total_saved} failed={total_failed}")
    if failed_ids:
        print(f"  {len(failed_ids)} cutout failures")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
