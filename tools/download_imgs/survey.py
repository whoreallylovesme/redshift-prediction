"""Shared SDSS coordinate lookup, Legacy Survey brick selection and streamed FITS downloads."""

from __future__ import annotations

import asyncio
import time
from io import StringIO
from pathlib import Path
from shutil import copyfileobj
from typing import TYPE_CHECKING
from urllib.parse import urlencode
from urllib.request import urlopen

import aiohttp
import numpy as np
import pandas as pd
from astropy.io import fits

if TYPE_CHECKING or __package__:
    from tools.download_imgs.workspace import download_destination
else:
    from workspace import download_destination

STAMP_SIZE = 224

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
        with urlopen(BRICKS_URL, timeout=120) as response, download_destination(local_path) as dest:
            copyfileobj(response, dest)
            length = response.headers.get("Content-Length")
            if length is not None and dest.tell() != int(length):
                raise ValueError("Truncated Legacy Survey brick index")
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
    counts = [len(objects) for objects in brick_map.values()]
    if counts:
        print(
            f"  {len(coords)} objects → {len(counts)} unique bricks "
            f"(avg {np.mean(counts):.0f} objects/brick, max {max(counts)}/brick)",
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
                    with download_destination(dest) as stream:
                        async for chunk in resp.content.iter_chunked(1 << 20):
                            stream.write(chunk)
                    return True
            except (aiohttp.ClientError, TimeoutError) as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"    FAIL {dest.name} — {e}", flush=True)
                    return False
                await asyncio.sleep(RETRY_BACKOFF * (2**attempt))
    return False
