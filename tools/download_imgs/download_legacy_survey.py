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
import time
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
import astropy.units as u
import click
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS

if TYPE_CHECKING or __package__:
    from tools.download_imgs import survey
    from tools.download_imgs.workspace import brick_workspace
else:
    import survey
    from workspace import brick_workspace

# Historical imports used by callers of the standalone download scripts.
BANDS = survey.BANDS
STAMP_SIZE = survey.STAMP_SIZE
_brick_url = survey._brick_url
fetch_coordinates = survey.fetch_coordinates
find_brick = survey.find_brick
load_brick_index = survey.load_brick_index
map_objects_to_bricks = survey.map_objects_to_bricks
_download_file = survey._download_file

PIXSCALE = 0.262  # arcsec/pixel (native LS DR10 optical)
_JSON = click.Path(exists=True, dir_okay=False, path_type=Path)
_OUT = click.Path(file_okay=False, dir_okay=True, path_type=Path)


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
    if not brick_map:
        print("No coordinates matched the survey brick index.")
        return

    # ── Step 4: Download bricks & extract cutouts ─────────────────────────────
    print(f"\n[4/4] Downloading {len(brick_map)} bricks from NERSC & extracting cutouts...")
    with brick_workspace(outdir, keep=keep_bricks) as brick_dir:
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

    elapsed = time.time() - t0
    print(f"\nDone! saved={total_saved} failed={total_failed} time={elapsed:.0f}s")
    if all_failed_ids:
        print(f"  {len(all_failed_ids)} failed cutout extractions")
    if missing_brick_ids:
        print(f"  {len(missing_brick_ids)} objects in missing bricks (not covered by DR10)")


if __name__ == "__main__":
    main()
