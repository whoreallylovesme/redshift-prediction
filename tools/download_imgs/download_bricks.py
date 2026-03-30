#!/usr/bin/env python3
"""Download Legacy Survey DR10 cutouts via NERSC brick coadds (streaming).

For each brick: download 6 FITS → extract cutouts → delete FITS.
Disk usage: only .npz outputs + 1 brick at a time (~72 MB).

Usage:
  python download_bricks.py --dataset teddy --split A
  python download_bricks.py --dataset happy --split A,B
  python download_bricks.py --dataset teddy,happy --split A,B,C,D
"""
from __future__ import annotations

import asyncio
import json
import shutil
import time
from io import StringIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import aiohttp
import astropy.units as u
import click
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

# ── Config ────────────────────────────────────────────────────────────────────
STAMP_SIZE = 224
NERSC_BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10"
BRICKS_URL = f"{NERSC_BASE}/survey-bricks.fits.gz"
BANDS = ["g", "r", "i", "z", "W1", "W2"]

# ── SDSS ──────────────────────────────────────────────────────────────────────
SDSS_DR = 19
SDSS_BATCH = 200
SDSS_TIMEOUT = 60.0
SDSS_ENDPOINT = f"https://skyserver.sdss.org/dr{SDSS_DR}/SkyServerWS/SearchTools/SqlSearch"

MAX_RETRIES = 5
RETRY_BACKOFF = 1.0

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


# ── Brick index ───────────────────────────────────────────────────────────────

def load_brick_index(cache_dir: Path) -> pd.DataFrame:
    local_path = cache_dir / "survey-bricks.fits.gz"
    if not local_path.exists():
        print("  Downloading brick index...", flush=True)
        with urlopen(BRICKS_URL, timeout=120) as r:  # noqa: S310
            local_path.write_bytes(r.read())
        print(f"  Saved ({local_path.stat().st_size // 1024 // 1024} MB)", flush=True)
    else:
        print("  Using cached brick index", flush=True)

    with fits.open(local_path) as hdul:
        data = hdul[1].data
        return pd.DataFrame({
            "brickname": [s.strip() for s in data["BRICKNAME"].astype(str)],
            "ra": data["RA"].astype(float),
            "dec": data["DEC"].astype(float),
            "ra1": data["RA1"].astype(float),
            "ra2": data["RA2"].astype(float),
            "dec1": data["DEC1"].astype(float),
            "dec2": data["DEC2"].astype(float),
        })


def find_brick(ra: float, dec: float, bricks: pd.DataFrame) -> str | None:
    mask = (
        (bricks["ra1"] <= ra)
        & (ra < bricks["ra2"])
        & (bricks["dec1"] <= dec)
        & (dec < bricks["dec2"])
    )
    matches = bricks.loc[mask, "brickname"]
    if len(matches) == 0:
        return None
    return str(matches.iloc[0])


def map_objects_to_bricks(
    coords: pd.DataFrame,
    bricks: pd.DataFrame,
) -> dict[str, list[tuple[int, float, float]]]:
    dec_min, dec_max = coords["dec"].min() - 0.5, coords["dec"].max() + 0.5
    bricks_filtered = bricks[(bricks["dec2"] >= dec_min) & (bricks["dec1"] <= dec_max)]
    print(
        f"  Filtered to {len(bricks_filtered)} bricks in Dec [{dec_min:.1f}, {dec_max:.1f}]",
        flush=True,
    )

    brick_map: dict[str, list[tuple[int, float, float]]] = {}
    no_brick = 0
    for row in coords.itertuples(index=False):
        brick = find_brick(float(row.ra), float(row.dec), bricks_filtered)
        if brick is None:
            no_brick += 1
            continue
        brick_map.setdefault(brick, []).append((int(row.objid), float(row.ra), float(row.dec)))

    if no_brick > 0:
        print(f"  WARNING: {no_brick} objects not in any brick", flush=True)
    objs_per_brick = [len(v) for v in brick_map.values()]
    print(
        f"  {len(coords)} objects → {len(brick_map)} unique bricks "
        f"(avg {np.mean(objs_per_brick):.0f}/brick, max {np.max(objs_per_brick)}/brick)",
        flush=True,
    )
    return brick_map


# ── Download one brick ────────────────────────────────────────────────────────

def _brick_url(brickname: str, band: str) -> str:
    aaa = brickname[:3]
    sign = 1 if "p" in brickname else -1
    dec_str = brickname.split("p")[-1] if "p" in brickname else brickname.split("m")[-1]
    dec_val = sign * float(dec_str) / 10.0
    region = "north" if dec_val >= 32.375 else "south"
    fname = f"legacysurvey-{brickname}-image-{band}.fits.fz"
    return f"{NERSC_BASE}/{region}/coadd/{aaa}/{brickname}/{fname}"


async def _download_file(
    session: aiohttp.ClientSession,
    url: str,
    dest: Path,
    semaphore: asyncio.Semaphore,
) -> bool:
    if dest.exists():
        return True
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.get(url) as resp:
                    if resp.status == 404:
                        return False
                    resp.raise_for_status()
                    data = await resp.read()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(data)
                    return True
            except (aiohttp.ClientError, TimeoutError) as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"    FAIL {dest.name} — {e}", flush=True)
                    return False
                await asyncio.sleep(RETRY_BACKOFF * (2**attempt))
    return False


async def download_one_brick(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    brickname: str,
    brick_dir: Path,
) -> Path | None:
    """Download all 6 band files for one brick. Returns brick path or None."""
    aaa = brickname[:3]
    dest_dir = brick_dir / aaa / brickname
    dest_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for band in BANDS:
        url = _brick_url(brickname, band)
        dest = dest_dir / f"legacysurvey-{brickname}-image-{band}.fits.fz"
        tasks.append(_download_file(session, url, dest, semaphore))

    results = await asyncio.gather(*tasks)
    if all(results):
        return dest_dir
    return None


# ── Cutout extraction ─────────────────────────────────────────────────────────

def extract_cutouts_from_brick(
    brick_path: Path,
    brickname: str,
    objects: list[tuple[int, float, float]],
    objid_to_dest: dict[int, Path],
) -> tuple[int, int, list[int]]:
    """Extract cutouts for all objects in one brick. Returns (saved, failed, failed_ids)."""
    saved = 0
    failed = 0
    failed_ids: list[int] = []

    for objid, ra, dec in objects:
        dest_dir = objid_to_dest.get(objid)
        if dest_dir is None:
            continue
        npz_path = dest_dir / f"{objid}.npz"
        if npz_path.exists():
            saved += 1
            continue

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
        channels = []
        ok = True

        for band in BANDS:
            fpath = brick_path / f"legacysurvey-{brickname}-image-{band}.fits.fz"
            if not fpath.exists():
                ok = False
                break
            try:
                with fits.open(fpath, memmap=False) as hdul:
                    hdu = hdul[1] if len(hdul) > 1 else hdul[0]
                    wcs = WCS(hdu.header)
                    data = np.array(hdu.data, dtype=np.float32)
                    cutout = Cutout2D(
                        data,
                        coord,
                        (STAMP_SIZE, STAMP_SIZE),
                        wcs=wcs,
                        mode="partial",
                        fill_value=0.0,
                    )
                    channels.append(cutout.data)
            except Exception:
                ok = False
                break

        if ok and len(channels) == 6:
            image = np.stack(channels, axis=0)
            image = np.moveaxis(image, 0, -1)  # (6,H,W) → (H,W,6)
            if image.shape == (STAMP_SIZE, STAMP_SIZE, 6):
                np.savez_compressed(npz_path, image=image)
                saved += 1
                continue

        failed += 1
        failed_ids.append(objid)

    return saved, failed, failed_ids


# ── Streaming pipeline ────────────────────────────────────────────────────────

async def _process_one_brick(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    brickname: str,
    objects: list[tuple[int, float, float]],
    objid_to_dest: dict[int, Path],
    brick_dir: Path,
) -> tuple[int, int, list[int], list[int]]:
    """Download one brick, extract cutouts, delete brick."""
    brick_path = await download_one_brick(session, semaphore, brickname, brick_dir)

    if brick_path is None:
        missing = [oid for oid, _, _ in objects]
        return 0, len(objects), [], missing

    loop = asyncio.get_event_loop()
    saved, failed, failed_ids = await loop.run_in_executor(
        None, extract_cutouts_from_brick, brick_path, brickname, objects, objid_to_dest
    )
    shutil.rmtree(brick_path, ignore_errors=True)
    return saved, failed, failed_ids, []


async def streaming_pipeline(
    brick_map: dict[str, list[tuple[int, float, float]]],
    objid_to_dest: dict[int, Path],
    brick_dir: Path,
    concurrency: int,
    brick_parallel: int,
    timeout: float,
) -> tuple[int, int, list[int], list[int]]:
    """Download brick → extract cutouts → delete brick, N bricks in parallel."""
    semaphore = asyncio.Semaphore(concurrency)
    total_conns = concurrency * brick_parallel
    connector = aiohttp.TCPConnector(limit=total_conns, ttl_dns_cache=300)
    timeout_cfg = aiohttp.ClientTimeout(total=timeout, connect=30)

    total_bricks = len(brick_map)
    total_saved = 0
    total_failed = 0
    all_failed_ids: list[int] = []
    missing_brick_ids: list[int] = []
    done_count = 0
    t0 = time.time()

    brick_items = list(brick_map.items())

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as session:
        for chunk_start in range(0, total_bricks, brick_parallel):
            chunk = brick_items[chunk_start : chunk_start + brick_parallel]

            tasks = [
                _process_one_brick(session, semaphore, brickname, objects, objid_to_dest, brick_dir)
                for brickname, objects in chunk
            ]
            results = await asyncio.gather(*tasks)

            for saved, failed, failed_ids, missing_ids in results:
                total_saved += saved
                total_failed += failed
                all_failed_ids.extend(failed_ids)
                missing_brick_ids.extend(missing_ids)

            done_count += len(chunk)
            if done_count % 10 <= brick_parallel or done_count == total_bricks:
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total_bricks - done_count) / rate if rate > 0 else 0
                print(
                    f"  [{done_count}/{total_bricks}] "
                    f"saved={total_saved} failed={total_failed} "
                    f"{rate:.1f} brick/s "
                    f"ETA={eta / 60:.0f}min",
                    flush=True,
                )

    return total_saved, total_failed, all_failed_ids, missing_brick_ids


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
@click.option(
    "--dataset",
    default="teddy,happy",
    show_default=True,
    help="teddy, happy, or teddy,happy",
)
@click.option("--split", default="A,B,C,D", show_default=True, help="A, B, C, D or any combo")
@click.option("--concurrency", default=10, show_default=True, help="HTTP connections per brick")
@click.option(
    "--brick-parallel",
    default=4,
    show_default=True,
    help="Bricks to process in parallel",
)
@click.option("--max-per-split", default=None, type=int, help="Limit per split (testing)")
@click.option("--timeout", default=120.0, show_default=True)
@click.option("--dry-run", is_flag=True, help="Only fetch coords + count bricks, no download")
def main(
    teddy_json: Path,
    happy_json: Path,
    outdir: Path,
    dataset: str,
    split: str,
    concurrency: int,
    brick_parallel: int,
    max_per_split: int | None,
    timeout: float,
    dry_run: bool,
) -> None:
    selected_datasets = [d.strip().lower() for d in dataset.split(",")]
    selected_splits = [s.strip().upper() for s in split.split(",")]
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Datasets: {selected_datasets} | Splits: {selected_splits}")
    print(
        f"Concurrency: {concurrency} | Brick parallel: {brick_parallel} | "
        f"Output: {outdir.resolve()}",
    )

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
    print("\n[1/3] Fetching coordinates from SDSS...")
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

    # ── Step 2: Brick mapping ─────────────────────────────────────────────────
    print("\n[2/3] Mapping objects to bricks...")
    bricks = load_brick_index(outdir)
    brick_map = map_objects_to_bricks(coords, bricks)

    if dry_run:
        n_objs = len(objid_to_dest)
        n_bricks = len(brick_map)
        objs_per_brick = [len(v) for v in brick_map.values()]

        # .npz output: 224*224*6*4 bytes (float32) per object, ~50% compression
        raw_bytes_per_obj = STAMP_SIZE * STAMP_SIZE * 6 * 4
        npz_bytes_per_obj = raw_bytes_per_obj * 0.5
        npz_total_gb = n_objs * npz_bytes_per_obj / (1024**3)

        # Brick download (temporary): ~12 MB per band × 6 = ~72 MB per brick
        brick_peak_mb = 72  # one brick at a time

        # Time estimate from teddy A benchmark: ~0.4 brick/s at concurrency=10,
        # scales with brick_parallel
        brick_rate = 0.4 * brick_parallel
        est_hours = n_bricks / brick_rate / 3600

        # SDSS coordinate fetch: ~200 ids/batch, ~0.3s/batch
        sdss_batches = (n_objs + SDSS_BATCH - 1) // SDSS_BATCH
        sdss_min = sdss_batches * 0.3 / 60

        print(f"\n{'=' * 50}")
        print("  DRY RUN SUMMARY")
        print(f"{'=' * 50}")
        print(f"  Objects:        {n_objs:,}")
        print(f"  Unique bricks:  {n_bricks:,}")
        print(f"  Objs/brick:     avg={np.mean(objs_per_brick):.1f}, "
              f"median={np.median(objs_per_brick):.0f}, "
              f"max={np.max(objs_per_brick)}")
        print(f"  No coords:      {len(no_coords_ids):,}")
        print()
        print(f"  Output .npz:    ~{npz_total_gb:.1f} GB")
        print(
            f"  Peak disk:      ~{brick_peak_mb * brick_parallel} MB "
            f"({brick_parallel} bricks at a time)",
        )
        print()
        print(f"  SDSS fetch:     ~{sdss_min:.0f} min ({sdss_batches} batches)")
        print(f"  Download+cut:   ~{est_hours:.1f} h ({n_bricks} bricks @ ~{brick_rate} brick/s)")
        print(f"  Total est:      ~{sdss_min / 60 + est_hours:.1f} h")
        print(f"{'=' * 50}")
        return

    # ── Step 3: Stream download + extract ─────────────────────────────────────
    brick_dir = outdir / "_tmp_bricks"
    brick_dir.mkdir(parents=True, exist_ok=True)

    print("\n[3/3] Streaming: download → extract → delete...")
    total_saved, total_failed, failed_ids, missing_ids = asyncio.run(
        streaming_pipeline(
            brick_map,
            objid_to_dest,
            brick_dir,
            concurrency,
            brick_parallel,
            timeout,
        )
    )

    # Cleanup temp dir
    shutil.rmtree(brick_dir, ignore_errors=True)

    # ── Report ────────────────────────────────────────────────────────────────
    report = {
        "total_objects": len(objid_to_dest),
        "saved": total_saved,
        "failed_no_coords": no_coords_ids,
        "failed_cutout": failed_ids,
        "failed_missing_brick": missing_ids,
        "failed_total": total_failed + len(no_coords_ids),
    }
    report_path = outdir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"\nDone! saved={total_saved} failed={total_failed}")
    if failed_ids:
        print(f"  {len(failed_ids)} cutout extraction failures")
    if missing_ids:
        print(f"  {len(missing_ids)} objects in missing bricks (not covered by DR10)")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
