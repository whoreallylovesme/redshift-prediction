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
import sys
from itertools import batched
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
from tqdm import tqdm

if TYPE_CHECKING or __package__:
    from tools.download_imgs import survey
    from tools.download_imgs.workspace import brick_workspace, download_destination
else:
    import survey
    from workspace import brick_workspace, download_destination

# Historical imports used by callers of the standalone download scripts.
BANDS = survey.BANDS
STAMP_SIZE = survey.STAMP_SIZE
_brick_url = survey._brick_url
SDSS_BATCH = survey.SDSS_BATCH
fetch_coordinates = survey.fetch_coordinates
find_brick = survey.find_brick
load_brick_index = survey.load_brick_index
map_objects_to_bricks = survey.map_objects_to_bricks
_download_file = survey._download_file

_OUT = click.Path(file_okay=False, dir_okay=True, path_type=Path)


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
                with download_destination(npz_path) as stream:
                    np.savez_compressed(stream, image=image)
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
    with brick_workspace(brick_dir) as scratch:
        brick_path = await download_one_brick(session, semaphore, brickname, scratch)

        if brick_path is None:
            missing = [oid for oid, _, _ in objects]
            return 0, len(objects), [], missing

        saved, failed, failed_ids = await asyncio.to_thread(
            extract_cutouts_from_brick, brick_path, brickname, objects, objid_to_dest
        )
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

    total_saved = 0
    total_failed = 0
    all_failed_ids: list[int] = []
    missing_brick_ids: list[int] = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as session:
        with tqdm(total=len(brick_map), unit="brick", desc="Cutouts", file=sys.stdout) as progress:
            for chunk in batched(brick_map.items(), brick_parallel, strict=False):
                results = await asyncio.gather(*(
                    _process_one_brick(
                        session, semaphore, brickname, objects, objid_to_dest, brick_dir
                    )
                    for brickname, objects in chunk
                ))
                for saved, failed, failed_ids, missing_ids in results:
                    total_saved += saved
                    total_failed += failed
                    all_failed_ids.extend(failed_ids)
                    missing_brick_ids.extend(missing_ids)
                progress.set_postfix(saved=total_saved, failed=total_failed, refresh=False)
                progress.update(len(chunk))

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
    if not brick_map:
        print("No coordinates matched the survey brick index.")
        return

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
    with brick_workspace(outdir) as brick_dir:
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
