#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path
from urllib.request import urlopen

import click
import numpy as np

DEEPDIP_URL = "https://deepdip.iap.fr/api/v1/file/60ef1e05be2b8ebb048d951e/download"
DEFAULT_NPZ = "sdss_pasquet.npz"


def load_ids(json_path: Path) -> dict[str, list[int]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {split: [int(x) for x in ids] for split, ids in data.items()}


def download_with_progress(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    resp = urlopen(url, timeout=300)  # noqa: S310
    total = int(resp.headers.get("Content-Length", 0))
    total_gb = total / (1 << 30) if total else 0
    downloaded = 0
    t0 = time.time()
    with open(dest, "wb") as f:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = time.time() - t0
            speed_mb = (downloaded / (1 << 20)) / elapsed if elapsed > 0 else 0
            pct = (downloaded / total * 100) if total else 0
            eta = (total - downloaded) / (downloaded / elapsed) if downloaded > 0 and total else 0
            print(
                f"\r  {downloaded / (1 << 30):.2f}/{total_gb:.2f} GB "
                f"({pct:.1f}%) {speed_mb:.1f} MB/s ETA {eta / 60:.0f}min",
                end="",
                flush=True,
            )
    print(f"\n  Done in {(time.time() - t0) / 60:.1f} min")


_JSON = click.Path(exists=True, dir_okay=False, path_type=Path)
_OUT = click.Path(file_okay=False, dir_okay=True, path_type=Path)


@click.command()
@click.option("--sdss-npz", default=DEFAULT_NPZ, type=click.Path(dir_okay=False, path_type=Path))
@click.option("--teddy-json", default="teddy_ids.json", type=_JSON)
@click.option("--happy-json", default="happy_ids.json", type=_JSON)
@click.option("--outdir", default="galaxy_images", type=_OUT)
@click.option("--skip-existing/--overwrite", default=True)
def main(
    sdss_npz: Path,
    teddy_json: Path,
    happy_json: Path,
    outdir: Path,
    skip_existing: bool,
) -> None:
    if not sdss_npz.exists():
        print(f"File {sdss_npz} not found. Downloading from DeepDIP (~50 GB)...")
        download_with_progress(DEEPDIP_URL, sdss_npz)
    else:
        print(f"Using existing file: {sdss_npz} ({sdss_npz.stat().st_size / (1 << 30):.1f} GB)")

    datasets = {"teddy": load_ids(teddy_json), "happy": load_ids(happy_json)}

    all_needed: dict[int, list[tuple[str, str]]] = {}
    for ds_name, splits in datasets.items():
        for split, ids in splits.items():
            for oid in ids:
                all_needed.setdefault(oid, []).append((ds_name, split))

    print(f"\nNeeded: {len(all_needed)} unique objIDs across Teddy/Happy")

    print("Loading Pasquet labels (memory-mapped)...")
    t0 = time.time()
    data = np.load(sdss_npz, mmap_mode="r", allow_pickle=True)
    pasquet_lookup = {int(oid): idx for idx, oid in enumerate(data["labels"]["objID"])}
    print(f"  Pasquet dataset: {len(pasquet_lookup)} galaxies (loaded in {time.time() - t0:.1f}s)")

    matched_ids = set(all_needed.keys()) & set(pasquet_lookup.keys())
    unmatched_ids = set(all_needed.keys()) - matched_ids
    print(f"  Matched: {len(matched_ids)} ({len(matched_ids) / len(all_needed) * 100:.1f}%)")
    print(f"  Unmatched: {len(unmatched_ids)}")

    print("\nExtracting images...")
    cube = data["cube"]

    for ds_name in ["teddy", "happy"]:
        for split in datasets[ds_name]:
            (outdir / ds_name / split).mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    t0 = time.time()

    for i, oid in enumerate(sorted(matched_ids, key=lambda oid: pasquet_lookup[oid])):
        idx = pasquet_lookup[oid]
        first_dest = None
        for ds_name, split in all_needed[oid]:
            npz_path = outdir / ds_name / split / f"{oid}.npz"
            if skip_existing and npz_path.exists():
                skipped += 1
                continue
            if first_dest is None:
                img = np.array(cube[idx], dtype=np.float32)
                np.savez_compressed(npz_path, image=img)
                first_dest = npz_path
                saved += 1
            else:
                os.link(first_dest, npz_path)
                saved += 1

        if (i + 1) % 10000 == 0 or i + 1 == len(matched_ids):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(matched_ids) - i - 1) / rate if rate > 0 else 0
            print(
                f"  {i + 1}/{len(matched_ids)} saved={saved} skipped={skipped} "
                f"rate={rate:.0f}obj/s ETA={eta / 60:.1f}min",
                flush=True,
            )

    print(f"\nDone: saved={saved} skipped={skipped} time={time.time() - t0:.1f}s")

    if unmatched_ids:
        unmatched_path = outdir / "unmatched_objids.json"
        with open(unmatched_path, "w") as f:
            json.dump(sorted(unmatched_ids), f)
        print(f"\n{len(unmatched_ids)} unmatched objIDs saved to {unmatched_path}")
    else:
        print("\nAll objects matched!")


if __name__ == "__main__":
    main()
