#!/usr/bin/env python3
"""Download Legacy Survey DR10 cutouts (griz + W1W2) via NERSC brick coadds.

Instead of hitting the rate-limited cutout API, this script:
  1. Fetches (ra, dec) from SDSS DR19 for each objid
  2. Downloads the survey-bricks index to map objects → bricks
  3. Downloads full coadd bricks from NERSC (no rate limit)
  4. Extracts 224×224 cutouts locally with astropy WCS

Output: {outdir}/{dataset}/{split}/{objid}.npz  →  image: (224, 224, 6) float32
        channels order: g, r, i, z, w1, w2
"""

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
PIXSCALE = 0.262  # arcsec/pixel (native LS DR10 optical)

NERSC_BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10"
BRICKS_URL = f"{NERSC_BASE}/survey-bricks.fits.gz"
BANDS = ["g", "r", "i", "z", "W1", "W2"]

# ── SDSS ──────────────────────────────────────────────────────────────────────
SDSS_DR = 19
SDSS_BATCH = 200
SDSS_TIMEOUT = 60.0
SDSS_ENDPOINT = f"https://skyserver.sdss.org/dr{SDSS_DR}/SkyServerWS/SearchTools/SqlSearch"

# ── Retry / concurrency ──────────────────────────────────────────────────────
MAX_RETRIES = 5
RETRY_BACKOFF = 1.0

_JSON = click.Path(exists=True, dir_okay=False, path_type=Path)
_OUT = click.Path(file_okay=False, dir_okay=True, path_type=Path)


# ── SDSS coordinate fetch (reused from previous version) ─────────────────────

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
        print(f"    SDSS query failed (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF * (2**attempt))
    print("    All SDSS retries failed for this batch", flush=True)
    return pd.DataFrame()


def fetch_coordinates(
    objids: list[int],
    cache_path: Path,
) -> pd.DataFrame:
    """Return DataFrame with objid/ra/dec, fetching from SDSS and caching to parquet."""
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
            # periodic save
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
    """Download (if needed) and load survey-bricks.fits.gz → DataFrame."""
    local_path = cache_dir / "survey-bricks.fits.gz"
    if not local_path.exists():
        print(f"  Downloading brick index from {BRICKS_URL}...", flush=True)
        with urlopen(BRICKS_URL, timeout=120) as r:  # noqa: S310
            local_path.write_bytes(r.read())
        print(f"  Saved brick index ({local_path.stat().st_size // 1024 // 1024} MB)", flush=True)
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
    """Find the brick containing (ra, dec)."""
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
    """Return {brickname: [(objid, ra, dec), ...]}."""
    # Pre-filter bricks to relevant Dec range
    dec_min, dec_max = coords["dec"].min() - 0.5, coords["dec"].max() + 0.5
    bricks_filtered = bricks[(bricks["dec2"] >= dec_min) & (bricks["dec1"] <= dec_max)]
    print(
        f"  Filtered to {len(bricks_filtered)} bricks in Dec range "
        f"[{dec_min:.1f}, {dec_max:.1f}]",
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
        print(f"  WARNING: {no_brick} objects not found in any brick", flush=True)
    objs_per_brick = [len(v) for v in brick_map.values()]
    print(
        f"  {len(coords)} objects → {len(brick_map)} unique bricks "
        f"(avg {np.mean(objs_per_brick):.0f} obj/brick, "
        f"max {np.max(objs_per_brick)} obj/brick)",
        flush=True,
    )
    return brick_map


# ── Async brick download from NERSC ──────────────────────────────────────────

def _brick_url(brickname: str, band: str) -> str:
    aaa = brickname[:3]
    # Parse dec from brick name: e.g. "0571p007" → +7, "0571m007" → -7
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
                        print(f"    404 {dest.name}", flush=True)
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


async def download_bricks(
    bricknames: list[str],
    brick_dir: Path,
    concurrency: int,
    timeout: float,
) -> dict[str, Path]:
    """Download brick FITS files, return {brickname: brick_dir_path}."""
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300)
    timeout_cfg = aiohttp.ClientTimeout(total=timeout, connect=30)

    total_files = len(bricknames) * len(BANDS)
    done = 0
    failed_bricks: set[str] = set()
    t0 = time.time()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as session:
        for bi, brick in enumerate(bricknames):
            aaa = brick[:3]
            dest_dir = brick_dir / aaa / brick
            dest_dir.mkdir(parents=True, exist_ok=True)

            tasks = []
            for band in BANDS:
                url = _brick_url(brick, band)
                dest = dest_dir / f"legacysurvey-{brick}-image-{band}.fits.fz"
                tasks.append(_download_file(session, url, dest, semaphore))

            results = await asyncio.gather(*tasks)
            done += len(BANDS)
            if not all(results):
                failed_bricks.add(brick)

            if (bi + 1) % 10 == 0 or (bi + 1) == len(bricknames):
                elapsed = time.time() - t0
                rate = (bi + 1) / elapsed if elapsed > 0 else 0
                eta = (len(bricknames) - bi - 1) / rate if rate > 0 else 0
                print(
                    f"  bricks: {bi + 1}/{len(bricknames)} "
                    f"({done}/{total_files} files) "
                    f"failed={len(failed_bricks)} "
                    f"rate={rate:.1f}brick/s "
                    f"ETA={eta / 60:.0f}min",
                    flush=True,
                )

    result = {}
    for brick in bricknames:
        if brick not in failed_bricks:
            aaa = brick[:3]
            result[brick] = brick_dir / aaa / brick
    return result


# ── Cutout extraction ─────────────────────────────────────────────────────────

def extract_cutout(
    brick_path: Path,
    brickname: str,
    ra: float,
    dec: float,
    size: int = STAMP_SIZE,
) -> np.ndarray | None:
    """Extract (size, size, 6) float32 cutout from brick FITS files."""
    channels = []
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    for band in BANDS:
        fpath = brick_path / f"legacysurvey-{brickname}-image-{band}.fits.fz"
        if not fpath.exists():
            return None
        with fits.open(fpath, memmap=False) as hdul:
            # .fits.fz (fpack compressed): data is in HDU 1
            hdu = hdul[1] if len(hdul) > 1 else hdul[0]
            wcs = WCS(hdu.header)
            data = hdu.data
            if data is None:
                return None
            data = np.array(data, dtype=np.float32)

            try:
                cutout = Cutout2D(
                    data,
                    coord,
                    (size, size),
                    wcs=wcs,
                    mode="partial",
                    fill_value=0.0,
                )
                channels.append(cutout.data)
            except Exception:
                return None

    if len(channels) != 6:
        return None

    # Stack: (6, H, W) → (H, W, 6)
    image = np.stack(channels, axis=0)
    image = np.moveaxis(image, 0, -1)

    # Verify shape
    if image.shape != (size, size, 6):
        return None

    return image


def process_brick_cutouts(
    brick_path: Path,
    brickname: str,
    objects: list[tuple[int, float, float]],
    dest_dirs: dict[int, Path],
) -> tuple[int, int, list[int]]:
    """Extract cutouts for all objects in a brick. Returns (saved, failed, failed_ids)."""
    saved = 0
    failed = 0
    failed_ids: list[int] = []
    for objid, ra, dec in objects:
        dest_dir = dest_dirs.get(objid)
        if dest_dir is None:
            continue
        npz_path = dest_dir / f"{objid}.npz"
        if npz_path.exists():
            saved += 1
            continue
        image = extract_cutout(brick_path, brickname, ra, dec)
        if image is not None:
            np.savez_compressed(npz_path, image=image)
            saved += 1
        else:
            failed += 1
            failed_ids.append(objid)
    return saved, failed, failed_ids


# ── Main pipeline ─────────────────────────────────────────────────────────────

@click.command()
@click.option("--teddy-json", default="teddy_ids.json", type=_JSON)
@click.option("--happy-json", default="happy_ids.json", type=_JSON)
@click.option("--outdir", default="legacy_images", type=_OUT)
@click.option(
    "--concurrency",
    default=10,
    show_default=True,
    help="Concurrent downloads from NERSC",
)
@click.option("--max-per-split", default=None, type=int, help="Limit objects per split (testing)")
@click.option("--timeout", default=120.0, show_default=True, help="Per-file download timeout (s)")
@click.option("--datasets", default="teddy,happy", show_default=True)
@click.option("--splits", default="A,B,C,D", show_default=True)
@click.option("--keep-bricks", is_flag=True, help="Keep downloaded brick files after extraction")
def main(
    teddy_json: Path,
    happy_json: Path,
    outdir: Path,
    concurrency: int,
    max_per_split: int | None,
    timeout: float,
    datasets: str,
    splits: str,
    keep_bricks: bool,
) -> None:
    selected_datasets = [d.strip().lower() for d in datasets.split(",")]
    selected_splits = [s.strip().upper() for s in splits.split(",")]
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Concurrency: {concurrency} | stamp: {STAMP_SIZE}x{STAMP_SIZE} | pixscale: {PIXSCALE}")
    print(f"Datasets: {selected_datasets} | Splits: {selected_splits}")
    print(f"Output: {outdir.resolve()}")

    # ── Load IDs ──────────────────────────────────────────────────────────────
    dataset_ids: dict[str, dict[str, list[int]]] = {}
    if "teddy" in selected_datasets:
        raw = json.loads(teddy_json.read_text())
        dataset_ids["teddy"] = {
            s: [int(x) for x in ids]
            for s, ids in raw.items()
            if s in selected_splits
        }
    if "happy" in selected_datasets:
        raw = json.loads(happy_json.read_text())
        dataset_ids["happy"] = {
            s: [int(x) for x in ids]
            for s, ids in raw.items()
            if s in selected_splits
        }

    # ── Collect all objids and their dest dirs ────────────────────────────────
    all_objids: set[int] = set()
    objid_to_dest: dict[int, Path] = {}
    objid_to_dataset_split: dict[int, tuple[str, str]] = {}

    for ds_name, splits_dict in dataset_ids.items():
        for split, ids in splits_dict.items():
            ids = ids[:max_per_split] if max_per_split else ids
            split_dir = outdir / ds_name / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for oid in ids:
                npz = split_dir / f"{oid}.npz"
                if not npz.exists():
                    all_objids.add(oid)
                    objid_to_dest[oid] = split_dir
                    objid_to_dataset_split[oid] = (ds_name, split)

    if not all_objids:
        print("All objects already downloaded!")
        return

    print(f"\n{'=' * 60}")
    print(f"Total objects to download: {len(all_objids)}")
    print(f"{'=' * 60}")

    # ── Step 1: Fetch coordinates ─────────────────────────────────────────────
    print("\n[1/4] Fetching coordinates from SDSS...")
    cache_path = outdir / "all_coords.parquet"
    coords = fetch_coordinates(list(all_objids), cache_path)
    if coords.empty:
        print("No coordinates found!")
        return

    # Filter to only objects we need
    coords = coords[coords["objid"].isin(all_objids)]
    print(f"  Got coordinates for {len(coords)}/{len(all_objids)} objects")

    # ── Step 2: Load brick index ──────────────────────────────────────────────
    print("\n[2/4] Loading brick index...")
    bricks = load_brick_index(outdir)
    print(f"  Total bricks in index: {len(bricks)}")

    # ── Step 3: Map objects to bricks ─────────────────────────────────────────
    print("\n[3/4] Mapping objects to bricks...")
    brick_map = map_objects_to_bricks(coords, bricks)

    # ── Step 4: Download bricks & extract cutouts ─────────────────────────────
    print(f"\n[4/4] Downloading {len(brick_map)} bricks from NERSC & extracting cutouts...")
    brick_dir = outdir / "bricks"
    brick_dir.mkdir(parents=True, exist_ok=True)

    brick_paths = asyncio.run(
        download_bricks(list(brick_map.keys()), brick_dir, concurrency, timeout)
    )
    print(f"  Downloaded {len(brick_paths)}/{len(brick_map)} bricks successfully")

    # ── Extract cutouts ───────────────────────────────────────────────────────
    print("\nExtracting cutouts...")
    total_saved = 0
    total_failed = 0
    all_failed_ids: list[int] = []
    missing_brick_ids: list[int] = []
    t0 = time.time()

    for bi, (brickname, objects) in enumerate(brick_map.items()):
        brick_path = brick_paths.get(brickname)
        if brick_path is None:
            total_failed += len(objects)
            missing_brick_ids.extend([oid for oid, _, _ in objects])
            continue

        saved, failed, failed_ids = process_brick_cutouts(
            brick_path,
            brickname,
            objects,
            objid_to_dest,
        )
        total_saved += saved
        total_failed += failed
        all_failed_ids.extend(failed_ids)

        if (bi + 1) % 50 == 0 or (bi + 1) == len(brick_map):
            elapsed = time.time() - t0
            rate = (total_saved + total_failed) / elapsed if elapsed > 0 else 0
            print(
                f"  bricks: {bi + 1}/{len(brick_map)} | "
                f"saved={total_saved} failed={total_failed} | "
                f"rate={rate:.0f}obj/s | "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    # ── Save report ──────────────────────────────────────────────────────────
    report = {
        "total_objects": len(all_objids),
        "saved": total_saved,
        "failed_cutout": all_failed_ids,
        "failed_missing_brick": missing_brick_ids,
        "failed_total": total_failed,
    }
    report_path = outdir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved to {report_path}", flush=True)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if not keep_bricks and brick_dir.exists():
        print("\nCleaning up brick files...")
        shutil.rmtree(brick_dir)

    elapsed = time.time() - t0
    print(f"\nDone! saved={total_saved} failed={total_failed} time={elapsed:.0f}s")
    if all_failed_ids:
        print(f"  {len(all_failed_ids)} failed cutout extractions")
    if missing_brick_ids:
        print(f"  {len(missing_brick_ids)} objects in missing bricks (not covered by DR10)")


if __name__ == "__main__":
    main()
