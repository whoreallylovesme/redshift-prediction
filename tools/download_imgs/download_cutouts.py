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
import time
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import aiohttp
import click
import numpy as np
import pandas as pd
from astropy.io import fits

# ── Config ────────────────────────────────────────────────────────────────────
STAMP_SIZE = 224
PIXSCALE = 0.262  # arcsec/pixel
BANDS = ["g", "r", "i", "z"]
CUTOUT_URL = "https://www.legacysurvey.org/viewer/cutout.fits"

# ── SDSS ──────────────────────────────────────────────────────────────────────
SDSS_DR = 19
SDSS_BATCH = 200
SDSS_TIMEOUT = 60.0
SDSS_ENDPOINT = f"https://skyserver.sdss.org/dr{SDSS_DR}/SkyServerWS/SearchTools/SqlSearch"

MAX_RETRIES = 8
RETRY_BACKOFF = 2.0

_OUT = click.Path(file_okay=False, dir_okay=True, path_type=Path)


# ── SDSS coordinate fetch ────────────────────────────────────────────────────

def _try_sdss(sql: str) -> pd.DataFrame | None:
    url = f"{SDSS_ENDPOINT}?{urlencode({'cmd': sql, 'format': 'csv'})}"
    try:
        with urlopen(url, timeout=SDSS_TIMEOUT) as r:  # noqa: S310
            text = r.read().decode("utf-8")
        if "No objects have been found" in text or "error" in text[:200].lower():
            return pd.DataFrame()
        df = pd.read_csv(StringIO(text), comment="#")
        if {"objid", "ra", "dec"}.issubset(set(df.columns)):
            return df
        return None
    except Exception:
        return None


def _sdss_batch(objids: list[int]) -> pd.DataFrame:
    sql = f"SELECT objid,ra,dec FROM PhotoObj WHERE objid IN ({','.join(map(str, objids))})"
    for attempt in range(MAX_RETRIES):
        result = _try_sdss(sql)
        if result is not None:
            return result
        print(f"    SDSS failed (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF * (2**attempt))
    print("    All retries failed for this batch", flush=True)
    return pd.DataFrame()


def fetch_coordinates(objids: list[int], cache_path: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cached_ids: set[int] = set()

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached_ids = set(cached["objid"].tolist())
        frames.append(cached)
        print(f"  Loaded {len(cached)} cached coordinates", flush=True)

    remaining = [oid for oid in objids if oid not in cached_ids]
    if remaining:
        print(f"  Querying SDSS for {len(remaining)} objects...", flush=True)
        batches = [remaining[i : i + SDSS_BATCH] for i in range(0, len(remaining), SDSS_BATCH)]
        new_frames: list[pd.DataFrame] = []
        for bi, batch in enumerate(batches):
            if bi % 50 == 0:
                print(f"    batch {bi + 1}/{len(batches)}", flush=True)
            df = _sdss_batch(batch)
            if not df.empty:
                new_frames.append(df)
            if (bi + 1) % 100 == 0 and new_frames:
                pd.concat(frames + new_frames, ignore_index=True).to_parquet(cache_path)
                print(f"    checkpoint saved ({bi + 1} batches done)", flush=True)
        if new_frames:
            frames.extend(new_frames)
        if frames:
            combined = pd.concat(frames, ignore_index=True).drop_duplicates("objid")
            combined.to_parquet(cache_path)
            print(f"  Cached {len(combined)} coordinates total", flush=True)

    if not frames:
        return pd.DataFrame(columns=["objid", "ra", "dec"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("objid")


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

                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dest_path, image=image)
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

    total = len(objects)
    saved = 0
    failed = 0
    failed_ids: list[int] = []
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as session:
        # Process in chunks to show progress
        chunk_size = 50
        for ci in range(0, total, chunk_size):
            chunk = objects[ci : ci + chunk_size]
            tasks = [
                download_one_cutout(session, semaphore, objid, ra, dec, dest)
                for objid, ra, dec, dest in chunk
            ]
            results = await asyncio.gather(*tasks)

            for (objid, _, _, _), ok in zip(chunk, results, strict=True):
                if ok:
                    saved += 1
                else:
                    failed += 1
                    failed_ids.append(objid)

            done = ci + len(chunk)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(
                f"  [{done}/{total}] saved={saved} failed={failed} "
                f"{rate:.2f} obj/s ETA={eta / 3600:.1f}h",
                flush=True,
            )

    return saved, failed, failed_ids


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
