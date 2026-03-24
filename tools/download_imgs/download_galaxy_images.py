#!/usr/bin/env python3
import bz2
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO, StringIO
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import click
import numpy as np
import pandas as pd
from astropy.io import fits

BANDS = ("u", "g", "r", "i", "z")
SKYVIEW_SURVEYS = ("SDSSu", "SDSSg", "SDSSr", "SDSSi", "SDSSz")
MAX_RETRIES = 5
RETRY_BACKOFF = 2.0
TIMEOUT = 180.0
SDSS_PIXEL_SCALE = 0.396

_skyview_semaphore: threading.Semaphore | None = None

_JSON = click.Path(exists=True, dir_okay=False, path_type=Path)
_OUT = click.Path(file_okay=False, dir_okay=True, path_type=Path)


def batched(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def request_bytes(url: str, timeout: float = TIMEOUT) -> bytes:
    for attempt in range(MAX_RETRIES):
        try:
            with urlopen(url, timeout=timeout) as resp:  # noqa: S310
                return cast(bytes, resp.read())
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if isinstance(exc, HTTPError) and exc.code == 404:
                raise
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_BACKOFF * (2**attempt))
    raise RuntimeError("unreachable")


def request_text(url: str) -> str:
    return request_bytes(url).decode("utf-8")


def load_ids(json_path: Path) -> dict[str, list[int]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return {split: [int(x) for x in ids] for split, ids in data.items()}


def fetch_metadata_batch(objids: list[int], dr: int, columns: str) -> pd.DataFrame:
    if not objids:
        return pd.DataFrame()
    sql = f"SELECT {columns} FROM PhotoObj WHERE objid IN ({','.join(str(x) for x in objids)})"
    base = f"https://skyserver.sdss.org/dr{dr}/SkyServerWS/SearchTools/SqlSearch"
    url = f"{base}?{urlencode({'cmd': sql, 'format': 'csv'})}"
    csv_text = request_text(url)
    if "No objects have been found" in csv_text:
        return pd.DataFrame()
    return pd.read_csv(StringIO(csv_text), comment="#")


def fetch_all_metadata(
    all_ids: list[int],
    dr: int,
    cache_path: Path,
    columns: str,
    batch_size: int = 200,
) -> pd.DataFrame:
    cached_ids: set[int] = set()
    frames: list[pd.DataFrame] = []

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached_ids = set(cached["objid"].tolist())
        frames.append(cached)
        print(f"  Loaded {len(cached)} cached metadata entries", flush=True)

    remaining = [oid for oid in all_ids if oid not in cached_ids]
    if not remaining:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    print(f"  Fetching metadata for {len(remaining)} objects...", flush=True)
    query_batches = batched(remaining, batch_size)
    new_frames: list[pd.DataFrame] = []
    failed_batches = 0

    for bi, chunk in enumerate(query_batches):
        if (bi + 1) % 20 == 0 or bi == 0 or bi + 1 == len(query_batches):
            print(f"    metadata batch {bi + 1}/{len(query_batches)}...", flush=True)
        try:
            df = fetch_metadata_batch(chunk, dr=dr, columns=columns)
            if not df.empty:
                new_frames.append(df)
        except Exception as exc:
            failed_batches += 1
            if failed_batches <= 3:
                print(f"    batch {bi + 1} failed: {exc}", flush=True)

        if (bi + 1) % 100 == 0 and new_frames:
            pd.concat(frames + new_frames, ignore_index=True).to_parquet(cache_path)

    if new_frames:
        frames.extend(new_frames)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(cache_path)
        print(f"  Cached {len(combined)} total metadata entries", flush=True)
        return combined
    return pd.DataFrame()


def _skyview_fetch_band(ra: float, dec: float, survey: str, stamp_size: int) -> np.ndarray:
    size_deg = stamp_size * SDSS_PIXEL_SCALE / 3600.0
    url = (
        f"https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl?"
        f"Position={ra},{dec}&Survey={survey}&Pixels={stamp_size}"
        f"&Size={size_deg}&Return=FITS&Projection=Tan&Sampler=Lanczos3"
    )
    for attempt in range(MAX_RETRIES):
        if _skyview_semaphore is not None:
            _skyview_semaphore.acquire()
        try:
            data = request_bytes(url, timeout=120)
        finally:
            if _skyview_semaphore is not None:
                _skyview_semaphore.release()
        if data[:6] == b"SIMPLE":
            with fits.open(BytesIO(data), memmap=False) as hdul:
                img = hdul[0].data
                if img is not None and img.size >= stamp_size * stamp_size:
                    return cast(np.ndarray, img.astype(np.float32))
        time.sleep(RETRY_BACKOFF * (2**attempt))
    raise ValueError(f"SkyView returned invalid data for {survey} at ({ra},{dec})")


def skyview_fetch_object(
    objid: int,
    ra: float,
    dec: float,
    stamp_size: int,
    dest_dir: Path,
) -> bool:
    npz_path = dest_dir / f"{objid}.npz"
    if npz_path.exists():
        return True
    with ThreadPoolExecutor(max_workers=len(SKYVIEW_SURVEYS)) as band_pool:
        futs = {
            band_pool.submit(_skyview_fetch_band, ra, dec, s, stamp_size): i
            for i, s in enumerate(SKYVIEW_SURVEYS)
        }
        channels: list[np.ndarray | None] = [None] * len(SKYVIEW_SURVEYS)
        for fut in as_completed(futs):
            channels[futs[fut]] = fut.result()
    np.savez_compressed(npz_path, image=np.stack(channels, axis=-1))  # type: ignore[arg-type]
    return True


def build_frame_url(run: int, rerun: int, camcol: int, field: int, band: str, dr: int) -> str:
    survey = "boss" if dr <= 12 else "eboss"
    return (
        f"https://data.sdss.org/sas/dr{dr}/{survey}/photoObj/frames/"
        f"{rerun}/{run}/{camcol}/"
        f"frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"
    )


def download_frame(run: int, rerun: int, camcol: int, field: int, band: str, dr: int) -> np.ndarray:
    url = build_frame_url(run, rerun, camcol, field, band, dr)
    with fits.open(BytesIO(bz2.decompress(request_bytes(url))), memmap=False) as hdul:
        return cast(np.ndarray, hdul[0].data).astype(np.float32)


def extract_stamp(frame_data: np.ndarray, rowc: float, colc: float, stamp_size: int) -> np.ndarray:
    h, w = frame_data.shape
    half = stamp_size // 2
    rc, cc = int(round(rowc)), int(round(colc))
    r0, r1 = rc - half, rc - half + stamp_size
    c0, c1 = cc - half, cc - half + stamp_size
    stamp = np.zeros((stamp_size, stamp_size), dtype=np.float32)
    fr0, fr1 = max(0, r0), min(h, r1)
    fc0, fc1 = max(0, c0), min(w, c1)
    if fr0 < fr1 and fc0 < fc1:
        rslice = slice(fr0 - r0, fr0 - r0 + fr1 - fr0)
        cslice = slice(fc0 - c0, fc0 - c0 + fc1 - fc0)
        stamp[rslice, cslice] = frame_data[fr0:fr1, fc0:fc1]
    return stamp


def sas_process_frame(
    frame_key: tuple[int, int, int, int],
    group: pd.DataFrame,
    outdir_map: dict[int, Path],
    dr: int,
    stamp_size: int,
) -> tuple[int, int]:
    run, rerun, camcol, field = frame_key
    all_exist = all(
        outdir_map.get(int(row.objid), Path("/x")).joinpath(f"{int(row.objid)}.npz").exists()
        for row in group.itertuples(index=False)
        if int(row.objid) in outdir_map
    )
    if all_exist:
        return 0, len(group)
    band_data: dict[str, np.ndarray | None] = {}
    for band in BANDS:
        try:
            band_data[band] = download_frame(run, rerun, camcol, field, band, dr)
        except Exception:
            band_data[band] = None
    if any(v is None for v in band_data.values()):
        return 0, 0
    saved = 0
    for row in group.itertuples(index=False):
        objid = int(row.objid)
        dest = outdir_map.get(objid)
        if dest is None:
            continue
        npz_path = dest / f"{objid}.npz"
        if npz_path.exists():
            continue
        channels = []
        for b in BANDS:
            ch = extract_stamp(
                cast(np.ndarray, band_data[b]),
                float(row.rowc),
                float(row.colc),
                stamp_size,
            )
            channels.append(ch)
        np.savez_compressed(npz_path, image=np.stack(channels, axis=-1))
        saved += 1
    return saved, 0


def process_dataset_skyview(
    *,
    dataset_name: str,
    ids_by_split: dict[str, list[int]],
    outdir: Path,
    dr: int,
    stamp_size: int,
    meta_batch_size: int,
    workers: int,
    max_per_split: int | None,
    skip_existing: bool,
) -> None:
    objid_to_dest: dict[int, Path] = {}
    all_ids: list[int] = []

    for split, split_ids in ids_by_split.items():
        ids = split_ids[:max_per_split] if max_per_split else split_ids
        split_dir = outdir / dataset_name / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for oid in ids:
            if skip_existing and (split_dir / f"{oid}.npz").exists():
                continue
            objid_to_dest[oid] = split_dir
            all_ids.append(oid)

    unique_ids = list(set(all_ids))
    if not unique_ids:
        print(f"[{dataset_name}] All objects already downloaded!", flush=True)
        return

    print(f"[{dataset_name}] {len(unique_ids)} objects to download", flush=True)
    meta_path = outdir / f"{dataset_name}_meta.parquet"
    meta = fetch_all_metadata(
        unique_ids,
        dr=dr,
        cache_path=meta_path,
        columns="objid, ra, dec",
        batch_size=meta_batch_size,
    )
    if meta.empty:
        print(f"[{dataset_name}] No metadata found!", flush=True)
        return

    coord_map = {
        int(row.objid): (float(row.ra), float(row.dec))
        for row in meta.itertuples(index=False)
    }
    print(f"[{dataset_name}] Coordinates for {len(coord_map)} objects", flush=True)

    tasks = []
    for oid in unique_ids:
        if oid in coord_map and oid in objid_to_dest:
            ra, dec = coord_map[oid]
            tasks.append((oid, ra, dec, objid_to_dest[oid]))
    total = len(tasks)
    saved = 0
    failed = 0
    t0 = time.time()

    global _skyview_semaphore
    max_concurrent = max(workers * 5, 15)
    _skyview_semaphore = threading.Semaphore(max_concurrent)
    print(
        f"[{dataset_name}] Downloading {total} cutouts with {workers} workers "
        f"(max {max_concurrent} concurrent)...",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(skyview_fetch_object, oid, ra, dec, stamp_size, dest): oid
            for oid, ra, dec, dest in tasks
        }
        done_count = 0
        for fut in as_completed(futures):
            oid = futures[fut]
            done_count += 1
            try:
                fut.result()
                saved += 1
            except Exception as exc:
                failed += 1
                if failed <= 10:
                    print(f"    objid {oid} failed: {exc}", flush=True)
            if done_count % 50 == 0 or done_count == total:
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / rate if rate > 0 else 0
                print(
                    f"  [{dataset_name}] {done_count}/{total} saved={saved} failed={failed} "
                    f"rate={rate:.2f}obj/s ETA={eta / 3600:.1f}h",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(f"[{dataset_name}] Done: saved={saved} failed={failed} time={elapsed:.1f}s", flush=True)


def process_dataset_sas(
    *,
    dataset_name: str,
    ids_by_split: dict[str, list[int]],
    outdir: Path,
    dr: int,
    stamp_size: int,
    meta_batch_size: int,
    workers: int,
    max_per_split: int | None,
    skip_existing: bool,
) -> None:
    objid_to_dest: dict[int, Path] = {}
    all_ids: list[int] = []

    for split, split_ids in ids_by_split.items():
        ids = split_ids[:max_per_split] if max_per_split else split_ids
        split_dir = outdir / dataset_name / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for oid in ids:
            if skip_existing and (split_dir / f"{oid}.npz").exists():
                continue
            objid_to_dest[oid] = split_dir
            all_ids.append(oid)

    unique_ids = list(set(all_ids))
    if not unique_ids:
        print(f"[{dataset_name}] All objects already downloaded!", flush=True)
        return

    print(f"[{dataset_name}] {len(unique_ids)} objects to download", flush=True)
    sas_path = outdir / f"{dataset_name}_meta_sas.parquet"
    cols = "objid, run, rerun, camcol, field, rowc, colc"
    meta = fetch_all_metadata(
        unique_ids,
        dr=dr,
        cache_path=sas_path,
        columns=cols,
        batch_size=meta_batch_size,
    )
    if meta.empty:
        print(f"[{dataset_name}] No metadata found!", flush=True)
        return

    print(f"[{dataset_name}] Metadata: {len(meta)} found", flush=True)
    grouped = sorted(
        meta.groupby(["run", "rerun", "camcol", "field"]),
        key=lambda x: len(x[1]),
        reverse=True,
    )
    total_frames = len(grouped)
    total_saved = 0
    total_failed = 0
    t0 = time.time()
    print(
        f"[{dataset_name}] Processing {total_frames} frames with {workers} workers...",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for fk, grp in grouped:
            frame_key = cast(tuple[int, int, int, int], tuple(int(v) for v in fk))
            fut = pool.submit(sas_process_frame, frame_key, grp, objid_to_dest, dr, stamp_size)
            futures[fut] = (fk, len(grp))
        done_count = 0
        for fut in as_completed(futures):
            fk, n_objects = futures[fut]
            done_count += 1
            try:
                s, _ = fut.result()
                total_saved += s
            except Exception as exc:
                total_failed += n_objects
                if total_failed <= 10:
                    print(f"    frame {fk} failed: {exc}", flush=True)
            if done_count % 100 == 0 or done_count == total_frames:
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total_frames - done_count) / rate if rate > 0 else 0
                print(
                    f"  [{dataset_name}] frames {done_count}/{total_frames} "
                    f"saved={total_saved} failed={total_failed} rate={rate:.1f}fr/s "
                    f"ETA={eta / 60:.0f}min",
                    flush=True,
                )

    elapsed = time.time() - t0
    print(
        f"[{dataset_name}] Done: saved={total_saved} failed={total_failed} time={elapsed:.1f}s",
        flush=True,
    )


@click.command()
@click.option("--teddy-json", default="teddy_ids.json", type=_JSON)
@click.option("--happy-json", default="happy_ids.json", type=_JSON)
@click.option("--outdir", default="galaxy_images", type=_OUT)
@click.option("--backend", default="skyview", type=click.Choice(["skyview", "sas"]))
@click.option("--dr", default=17, type=int)
@click.option("--stamp-size", default=64, type=int)
@click.option("--meta-batch-size", default=200, type=int)
@click.option("--workers", default=10, type=int)
@click.option("--max-per-split", default=None, type=int)
@click.option("--skip-existing/--overwrite-existing", default=True)
@click.option("--unmatched-json", default=None, type=_JSON)
@click.option("--datasets", default="teddy,happy", show_default=True)
@click.option("--splits", default="A,B,C,D", show_default=True)
def main(
    teddy_json: Path,
    happy_json: Path,
    outdir: Path,
    backend: str,
    dr: int,
    stamp_size: int,
    meta_batch_size: int,
    workers: int,
    max_per_split: int | None,
    skip_existing: bool,
    unmatched_json: Path | None,
    datasets: str,
    splits: str,
) -> None:
    selected_datasets = [d.strip().lower() for d in datasets.split(",")]
    selected_splits = [s.strip().upper() for s in splits.split(",")]

    print(f"Backend: {backend} | DR{dr} | stamp={stamp_size}x{stamp_size} | workers={workers}")
    print(f"Datasets: {selected_datasets} | Splits: {selected_splits}")
    print(f"Output: {outdir.resolve()}")

    teddy_ids = load_ids(teddy_json)
    happy_ids = load_ids(happy_json)

    teddy_ids = {s: ids for s, ids in teddy_ids.items() if s in selected_splits}
    happy_ids = {s: ids for s, ids in happy_ids.items() if s in selected_splits}

    if unmatched_json is not None:
        unmatched_set = set(json.loads(unmatched_json.read_text()))
        print(f"Filtering to {len(unmatched_set)} unmatched objIDs from {unmatched_json}")
        teddy_ids = {s: [x for x in ids if x in unmatched_set] for s, ids in teddy_ids.items()}
        happy_ids = {s: [x for x in ids if x in unmatched_set] for s, ids in happy_ids.items()}

    process_fn = process_dataset_skyview if backend == "skyview" else process_dataset_sas

    for name, ids_by_split in [("teddy", teddy_ids), ("happy", happy_ids)]:
        if name not in selected_datasets:
            continue
        print(f"\n{'=' * 60}", flush=True)
        print(f"Dataset: {name}", flush=True)
        print(f"{'=' * 60}", flush=True)
        process_fn(
            dataset_name=name,
            ids_by_split=ids_by_split,
            outdir=outdir,
            dr=dr,
            stamp_size=stamp_size,
            meta_batch_size=meta_batch_size,
            workers=workers,
            max_per_split=max_per_split,
            skip_existing=skip_existing,
        )


if __name__ == "__main__":
    main()
