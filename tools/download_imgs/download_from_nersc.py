#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TypedDict

import click
import numpy as np


class NerscFileInfo(TypedDict):
    url: str
    size_gb: int


NERSC_FILES: dict[str, NerscFileInfo] = {
    "sdss_w_specz_train.h5": {
        "url": "https://portal.nersc.gov/project/dasrepo/self-supervised-learning-sdss/datasets/sdss_w_specz_train.h5",
        "size_gb": 85,
    },
    "sdss_w_specz_valid.h5": {
        "url": "https://portal.nersc.gov/project/dasrepo/self-supervised-learning-sdss/datasets/sdss_w_specz_valid.h5",
        "size_gb": 22,
    },
}

CROP_SIZE = 64


def load_ids(json_path: Path) -> dict[str, list[int]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {split: [int(x) for x in ids] for split, ids in data.items()}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for cmd in [
        ["wget", "-c", "--progress=bar:force:noscroll", "-O", str(dest), url],
        ["curl", "-L", "-C", "-", "-o", str(dest), "--progress-bar", url],
    ]:
        try:
            subprocess.run(cmd, check=True)
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            raise
    raise RuntimeError("Neither wget nor curl found.")


def center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0:y0 + size, x0:x0 + size]


def process_h5_file(
    h5_path: Path,
    all_needed: dict[int, list[tuple[str, str]]],
    outdir: Path,
    skip_existing: bool,
) -> tuple[int, int, set[int]]:
    import h5py

    saved = 0
    skipped = 0
    matched_here: set[int] = set()

    with h5py.File(h5_path, "r") as f:
        objids = f["ObjID"][:]
        images = f["images"]

        objid_to_idx = {int(oid): idx for idx, oid in enumerate(objids) if int(oid) in all_needed}
        print(f"  Found {len(objid_to_idx)} matching objects in {h5_path.name}")
        matched_here = set(objid_to_idx.keys())

        t0 = time.time()
        for i, (oid, idx) in enumerate(sorted(objid_to_idx.items(), key=lambda x: x[1])):
            first_dest = None
            for ds_name, split in all_needed[oid]:
                npz_path = outdir / ds_name / split / f"{oid}.npz"
                if skip_existing and npz_path.exists():
                    skipped += 1
                    continue
                if first_dest is None:
                    img = center_crop(np.array(images[idx], dtype=np.float32), CROP_SIZE)
                    np.savez_compressed(npz_path, image=img)
                    first_dest = npz_path
                    saved += 1
                else:
                    os.link(first_dest, npz_path)
                    saved += 1

            if (i + 1) % 10000 == 0 or i + 1 == len(objid_to_idx):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(objid_to_idx) - i - 1) / rate if rate > 0 else 0
                print(
                    f"    {i + 1}/{len(objid_to_idx)} saved={saved} skipped={skipped} "
                    f"rate={rate:.0f}obj/s ETA={eta / 60:.1f}min",
                    flush=True,
                )

    return saved, skipped, matched_here


_JSON = click.Path(exists=True, dir_okay=False, path_type=Path)
_OUT = click.Path(file_okay=False, dir_okay=True, path_type=Path)


@click.command()
@click.option("--data-dir", default=".", type=click.Path(file_okay=False, path_type=Path))
@click.option("--teddy-json", default="teddy_ids.json", type=_JSON)
@click.option("--happy-json", default="happy_ids.json", type=_JSON)
@click.option("--outdir", default="galaxy_images", type=_OUT)
@click.option("--skip-existing/--overwrite", default=True)
@click.option("--skip-download", is_flag=True)
@click.option("--count-only", is_flag=True)
def main(
    data_dir: Path,
    teddy_json: Path,
    happy_json: Path,
    outdir: Path,
    skip_existing: bool,
    skip_download: bool,
    count_only: bool,
) -> None:
    try:
        import h5py  # noqa: F401
    except ImportError:
        print("h5py is required: pip install h5py")
        sys.exit(1)

    data_dir.mkdir(parents=True, exist_ok=True)
    for fname, info in NERSC_FILES.items():
        fpath = data_dir / fname
        if fpath.exists():
            print(f"Found {fname} ({fpath.stat().st_size / (1 << 30):.1f} GB)")
        elif skip_download:
            print(f"WARNING: {fname} not found, skipping")
        else:
            print(f"\nDownloading {fname} (~{info['size_gb']} GB)...")
            download_file(info["url"], fpath)

    datasets = {"teddy": load_ids(teddy_json), "happy": load_ids(happy_json)}

    all_needed: dict[int, list[tuple[str, str]]] = {}
    for ds_name, splits in datasets.items():
        for split, ids in splits.items():
            for oid in ids:
                all_needed.setdefault(oid, []).append((ds_name, split))

    print(f"\nNeeded: {len(all_needed)} unique objIDs")

    if not count_only:
        for ds_name in ["teddy", "happy"]:
            for split in datasets[ds_name]:
                (outdir / ds_name / split).mkdir(parents=True, exist_ok=True)

    total_saved = 0
    total_skipped = 0
    all_matched: set[int] = set()
    t0 = time.time()

    for fname in NERSC_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            continue
        print(f"\n{'Scanning' if count_only else 'Processing'} {fname}...")
        if count_only:
            import h5py
            with h5py.File(fpath, "r") as f:
                matched = {int(oid) for oid in f["ObjID"][:] if int(oid) in all_needed}
            print(f"  Found {len(matched)} matching objects")
            saved, skipped = 0, 0
        else:
            saved, skipped, matched = process_h5_file(fpath, all_needed, outdir, skip_existing)
        total_saved += saved
        total_skipped += skipped
        all_matched |= matched

    unmatched = set(all_needed.keys()) - all_matched
    print(f"\n{'=' * 60}")
    pct = len(all_matched) / len(all_needed) * 100
    print(f"NERSC matched: {len(all_matched)}/{len(all_needed)} ({pct:.1f}%)")
    if not count_only:
        mins = (time.time() - t0) / 60
        print(f"Saved: {total_saved}  Skipped: {total_skipped}  Time: {mins:.1f} min")

    if unmatched:
        unmatched_path = outdir / "unmatched_objids.json"
        with open(unmatched_path, "w") as f:
            json.dump(sorted(unmatched), f)
        print(f"\n{len(unmatched)} unmatched objIDs saved to {unmatched_path}")
    else:
        print("\nAll objects matched!")


if __name__ == "__main__":
    main()
