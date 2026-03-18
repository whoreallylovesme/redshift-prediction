#!/usr/bin/env python3

import bz2
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
from pathlib import Path
from typing import cast
from urllib.parse import urlencode
from urllib.request import urlopen

import click
import numpy as np
import pandas as pd
from astropy.io import fits

BANDS = ("u", "g", "r", "i", "z")


def batched(items: list[int], size: int) -> list[list[int]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


TIMEOUT = 120.0


def request_bytes(url: str) -> bytes:
    with urlopen(url, timeout=TIMEOUT) as resp:  # noqa: S310
        return cast(bytes, resp.read())


def request_text(url: str) -> str:
    return request_bytes(url).decode("utf-8")


def load_ids(json_path: Path) -> dict[str, list[int]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {split: [int(x) for x in ids] for split, ids in data.items()}


def fetch_metadata(objids: list[int], dr: int) -> pd.DataFrame:
    if not objids:
        return pd.DataFrame()

    id_list = ",".join(str(x) for x in objids)
    sql = (
        "SELECT objid, run, rerun, camcol, field, rowc, colc "
        "FROM PhotoObj "
        f"WHERE objid IN ({id_list})"
    )
    endpoint = f"https://skyserver.sdss.org/dr{dr}/SkyServerWS/SearchTools/SqlSearch"
    url = f"{endpoint}?{urlencode({'cmd': sql, 'format': 'csv'})}"

    csv_text = request_text(url)
    if "No objects have been found" in csv_text:
        return pd.DataFrame()

    return pd.read_csv(StringIO(csv_text), comment="#")


def build_frame_url(
    run: int,
    rerun: int,
    camcol: int,
    field: int,
    band: str,
    dr: int,
) -> str:
    return (
        f"https://data.sdss.org/sas/dr{dr}/eboss/photoObj/frames/"
        f"{rerun}/{run}/{camcol}/"
        f"frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
    )


def download_frame(
    run: int,
    rerun: int,
    camcol: int,
    field: int,
    band: str,
    dr: int,
) -> np.ndarray:
    url = build_frame_url(run, rerun, camcol, field, band, dr)
    compressed = request_bytes(url)
    decompressed = bz2.decompress(compressed)
    with fits.open(BytesIO(decompressed), memmap=False) as hdul:
        return cast(np.ndarray, hdul[0].data).astype(np.float32)


def extract_stamp(
    frame_data: np.ndarray,
    rowc: float,
    colc: float,
    stamp_size: int,
) -> np.ndarray:
    h, w = frame_data.shape
    half = stamp_size // 2

    rc = int(round(rowc))
    cc = int(round(colc))

    r0 = rc - half
    r1 = rc - half + stamp_size
    c0 = cc - half
    c1 = cc - half + stamp_size

    stamp = np.zeros((stamp_size, stamp_size), dtype=np.float32)

    fr0, fr1 = max(0, r0), min(h, r1)
    fc0, fc1 = max(0, c0), min(w, c1)

    sr0 = fr0 - r0
    sr1 = sr0 + (fr1 - fr0)
    sc0 = fc0 - c0
    sc1 = sc0 + (fc1 - fc0)

    if fr0 < fr1 and fc0 < fc1:
        stamp[sr0:sr1, sc0:sc1] = frame_data[fr0:fr1, fc0:fc1]

    return stamp


def process_dataset(
    *,
    dataset_name: str,
    ids_by_split: dict[str, list[int]],
    outdir: Path,
    dr: int,
    stamp_size: int,
    batch_size: int,
    workers: int,
    max_per_split: int | None,
    skip_existing: bool,
) -> None:
    for split, split_ids in ids_by_split.items():
        ids = split_ids[:max_per_split] if max_per_split else split_ids
        split_dir = outdir / dataset_name / split
        split_dir.mkdir(parents=True, exist_ok=True)

        if skip_existing:
            ids = [oid for oid in ids if not (split_dir / f"{oid}.npz").exists()]

        if not ids:
            print(f"[{dataset_name}:{split}] nothing to download", flush=True)
            continue

        meta_frames: list[pd.DataFrame] = []
        batches = batched(ids, batch_size)
        for bi, chunk in enumerate(batches):
            print(
                f"  [{dataset_name}:{split}] metadata batch {bi + 1}/{len(batches)}...",
                flush=True,
            )
            df = fetch_metadata(chunk, dr=dr)
            if not df.empty:
                meta_frames.append(df)

        if not meta_frames:
            print(f"[{dataset_name}:{split}] no metadata found", flush=True)
            continue

        meta = pd.concat(meta_frames, ignore_index=True)
        print(
            f"[{dataset_name}:{split}] {len(ids)} requested, {len(meta)} found in SDSS",
            flush=True,
        )

        frame_key_cols = ["run", "rerun", "camcol", "field"]
        grouped = meta.groupby(frame_key_cols)
        total_frames = len(grouped)

        done = 0
        failed = 0

        for frame_idx, (frame_key, group) in enumerate(grouped):
            run, rerun, camcol, field = (int(v) for v in frame_key)

            band_data: dict[str, np.ndarray | None] = {}
            with ThreadPoolExecutor(max_workers=min(workers, len(BANDS))) as pool:
                futures = {
                    pool.submit(download_frame, run, rerun, camcol, field, band, dr): band
                    for band in BANDS
                }
                for fut in as_completed(futures):
                    band = futures[fut]
                    try:
                        band_data[band] = fut.result()
                    except Exception:  # noqa: BLE001
                        band_data[band] = None

            missing_bands = [b for b, v in band_data.items() if v is None]
            if missing_bands:
                print(
                    f"  frame ({run},{rerun},{camcol},{field}): "
                    f"failed bands {missing_bands}, skipping {len(group)} objects",
                    flush=True,
                )
                failed += len(group)
                continue

            for row in group.itertuples(index=False):
                objid = int(row.objid)
                rowc = float(row.rowc)
                colc = float(row.colc)

                channels = [
                    extract_stamp(cast(np.ndarray, band_data[band]), rowc, colc, stamp_size)
                    for band in BANDS
                ]
                tensor = np.stack(channels, axis=-1)  # (stamp_size, stamp_size, 5)

                np.savez_compressed(split_dir / f"{objid}.npz", image=tensor)
                done += 1

            if (frame_idx + 1) % 20 == 0 or frame_idx + 1 == total_frames:
                print(
                    f"  [{dataset_name}:{split}] frames {frame_idx + 1}/{total_frames}, "
                    f"saved={done}, failed={failed}",
                    flush=True,
                )

        print(f"[{dataset_name}:{split}] done: saved={done} failed={failed}", flush=True)


# https://data.sdss.org/sas/dr17/eboss/photoObj/frames/

@click.command()
@click.option(
    "--teddy-json",
    default="teddy_ids.json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--happy-json",
    default="happy_ids.json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--outdir",
    default="galaxy_images",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
)
@click.option("--dr", default=17, type=int)
@click.option("--stamp-size", default=64, type=int)
@click.option("--batch-size", default=200, type=int)
@click.option("--workers", default=5, type=int)
@click.option("--max-per-split", default=None, type=int)
@click.option("--skip-existing/--overwrite-existing", default=True)
def main(
    teddy_json: Path,
    happy_json: Path,
    outdir: Path,
    dr: int,
    stamp_size: int,
    batch_size: int,
    workers: int,
    max_per_split: int | None,
    skip_existing: bool,
) -> None:
    teddy_ids = load_ids(teddy_json)
    happy_ids = load_ids(happy_json)

    for name, ids_by_split in [("teddy", teddy_ids), ("happy", happy_ids)]:
        process_dataset(
            dataset_name=name,
            ids_by_split=ids_by_split,
            outdir=outdir,
            dr=dr,
            stamp_size=stamp_size,
            batch_size=batch_size,
            workers=workers,
            max_per_split=max_per_split,
            skip_existing=skip_existing,
        )


if __name__ == "__main__":
    main()
