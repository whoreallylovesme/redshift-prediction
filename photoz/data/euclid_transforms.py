from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from lib.utils.utils import check

BANDS = (
    "VIS",
    "Y",
    "J",
    "H",
    "U_EXT_MEGACAM",
    "G_EXT_HSC",
    "R_EXT_MEGACAM",
    "I_EXT_PANSTARRS",
    "Z_EXT_HSC",
    "U_EXT_DECAM",
    "G_EXT_DECAM",
    "R_EXT_DECAM",
    "I_EXT_DECAM",
    "Z_EXT_DECAM",
)
IMAGE_BANDS = 4
IMAGE_SIDE = 64
SPECTRUM_LENGTH = 531
UNIT_CGS = 1e-16

type Product = Mapping[str, NDArray[Any]]


class ImageNormalizer(BaseModel):
    scale_ujy: list[float]
    scale_fallback_RMS: list[bool]
    mean: list[float]
    std: list[float]
    observed_pixels: list[int]
    fit_population: str = "available supervised FTRAIN images only"
    scale_definition: str = (
        "exact median abs observed physical flux; positive RMS median fallback"
    )


class TableNormalizer(BaseModel):
    physical_scale_ujy: float = 1.0
    mean: list[float]
    std: list[float]
    pool: str = "all observed FTRAIN physical filters; same two scalars for every filter"


class SpectrumNormalizer(BaseModel):
    floor_cgs: float
    floor_fitted_on: str = "present SSL F/S bins only"
    per_object: str = "centre median;scale=max(1.4826MAD,medianRMS,floor)"
    metadata_unit_cgs: float = UNIT_CGS


class Normalizers(BaseModel):
    image: ImageNormalizer
    table: TableNormalizer
    spectrum: SpectrumNormalizer
    no_N_fitting: bool = True
    no_FVALID_fitting: bool = True


def load_normalizers(path: Path) -> Normalizers:
    return Normalizers.model_validate(json.loads(Path(path).read_text()))


def save_normalizers(state: Normalizers, path: Path) -> None:
    Path(path).write_text(json.dumps(state.model_dump(), indent=2) + "\n")


def validate_band_order(bands: Sequence[str] | NDArray[Any]) -> None:
    check(np.asarray(bands).tolist() == list(BANDS), "exact native14 physical band order")


def image_usable(a: Product) -> bool:
    mask = a["image_valid"]
    return bool(np.any(a["image_band_present"] & mask.reshape(IMAGE_BANDS, -1).any(1)))


def image_complete(a: Product) -> bool:
    return bool(
        np.all(a["image_band_present"])
        and np.all(a["image_valid"].reshape(IMAGE_BANDS, -1).any(1))
    )


def table_base(
    flux: NDArray[np.floating], error: NDArray[np.floating], valid: NDArray[np.bool_]
) -> NDArray[np.float64]:
    f = np.where(valid, flux, 0.0).astype(np.float64)
    e = np.where(valid, error, 0.0).astype(np.float64)
    return np.stack((np.arcsinh(f / 1.0), np.log1p(e / 1.0)), axis=-1)


def fit_table(
    flux: NDArray[np.floating], error: NDArray[np.floating], valid: NDArray[np.bool_]
) -> TableNormalizer:
    base = table_base(flux, error, valid)
    values = base[valid]
    check(len(values) > 0, "nonempty FTRAIN table observations")
    mean = values.mean(0)
    std = values.std(0)
    std[std <= 0] = 1
    return TableNormalizer(physical_scale_ujy=1.0, mean=mean.tolist(), std=std.tolist())


def transform_table(
    flux: NDArray[np.floating],
    error: NDArray[np.floating],
    valid: NDArray[np.bool_],
    state: TableNormalizer,
) -> NDArray[np.float32]:
    x = (table_base(flux, error, valid) - np.asarray(state.mean)) / np.asarray(state.std)
    x[~valid] = 0
    check(np.isfinite(x).all(), "finite table transform")
    out: NDArray[np.float32] = x.astype(np.float32)
    return out


def image_base(a: Product, scales: Sequence[float]) -> NDArray[np.float64]:
    mask = a["image_valid"]
    f = np.where(mask, a["image_flux_uJy"], 0.0).astype(np.float64)
    e = np.where(mask, a["image_rms_uJy"], 0.0).astype(np.float64)
    s = np.asarray(scales)[:, None, None]
    return np.concatenate((np.arcsinh(f / s), np.log1p(e / s)), axis=0)


def transform_image(a: Product, state: ImageNormalizer) -> NDArray[np.float32]:
    v = a["image_valid"]
    x = image_base(a, state.scale_ujy)
    x = (x - np.asarray(state.mean)[:, None, None]) / np.asarray(state.std)[:, None, None]
    x[~np.concatenate((v, v))] = 0
    result = np.concatenate((x, v.astype(np.float64)), axis=0).astype(np.float32)
    check(np.isfinite(result).all(), "finite image transform")
    return result


def transform_image_band(
    flux: NDArray[np.floating],
    error: NDArray[np.floating],
    valid: NDArray[np.bool_],
    state: ImageNormalizer,
    band: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    s = state.scale_ujy[band]
    with np.errstate(over="ignore", invalid="ignore"):
        f = np.where(valid, flux, 0.0).astype(np.float64)
        e = np.where(valid, error, 0.0).astype(np.float64)
        cf = (np.arcsinh(f / s) - state.mean[band]) / state.std[band]
        ce = (np.log1p(e / s) - state.mean[IMAGE_BANDS + band]) / state.std[IMAGE_BANDS + band]
    cf[~valid] = 0
    ce[~valid] = 0
    return cf, ce


def transform_spectrum(
    a: Product, floor: float
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32], bool]:
    valid = a["spectrum_valid"].copy()
    result = np.zeros((SPECTRUM_LENGTH, 4), np.float32)
    metadata = np.zeros(3, np.float32)
    if not bool(a["spectrum_present"]):
        return result, np.zeros(SPECTRUM_LENGTH, bool), metadata, False
    f = a["spectrum_flux_cgs"].astype(np.float64)
    e = a["spectrum_error_cgs"].astype(np.float64)
    w = a["spectrum_wavelength_angstrom"].astype(np.float64)
    centre = float(np.median(f[valid]))
    mad = float(1.4826 * np.median(np.abs(f[valid] - centre)))
    scale = max(mad, float(np.median(e[valid])), float(floor))
    check(np.isfinite(scale) and scale > 0, "positive spectrum scale")
    result[valid, 0] = ((f[valid] - centre) / scale).astype(np.float32)
    result[valid, 1] = (e[valid] / scale).astype(np.float32)
    result[valid, 2] = 1
    result[valid, 3] = np.log(w[valid] / 15000.0).astype(np.float32)
    metadata[:] = [np.arcsinh(centre / UNIT_CGS), np.log1p(scale / UNIT_CGS), valid.mean()]
    check(
        np.isfinite(result).all() and np.isfinite(metadata).all(), "finite spectrum transform"
    )
    return result, valid, metadata, True


def median_f32_inplace(values: NDArray[np.float32]) -> float:
    check(
        values.dtype == np.float32 and values.ndim == 1 and len(values) > 0,
        "nonempty f32 median",
    )
    k = len(values) // 2
    if len(values) % 2:
        values.partition(k)
        return float(values[k])
    values.partition((k - 1, k))
    return (float(values[k - 1]) + float(values[k])) / 2.0


def fit_image(
    ids: Sequence[int] | NDArray[np.integer], get_product: Callable[[int], Product]
) -> ImageNormalizer:
    buffer = np.empty(len(ids) * IMAGE_SIDE * IMAGE_SIDE, np.float32)
    scales: list[float] = []
    fallback: list[bool] = []
    for band in range(IMAGE_BANDS):
        used = 0
        for identifier in ids:
            a = get_product(int(identifier))
            v = a["image_valid"][band]
            values = np.abs(a["image_flux_uJy"][band][v])
            buffer[used : used + len(values)] = values
            used += len(values)
        check(used > 0, "FTRAIN measured pixels needed for every band")
        scale = median_f32_inplace(buffer[:used])
        use_rms = not np.isfinite(scale) or scale <= 0
        if use_rms:
            used = 0
            for identifier in ids:
                a = get_product(int(identifier))
                v = a["image_valid"][band]
                values = a["image_rms_uJy"][band][v]
                buffer[used : used + len(values)] = values
                used += len(values)
            scale = median_f32_inplace(buffer[:used])
        check(np.isfinite(scale) and scale > 0, "FTRAIN image positive scale")
        scales.append(scale)
        fallback.append(bool(use_rms))
    del buffer
    count = np.zeros(2 * IMAGE_BANDS, np.int64)
    sums = np.zeros(2 * IMAGE_BANDS, np.float64)
    squares = np.zeros(2 * IMAGE_BANDS, np.float64)
    for identifier in ids:
        a = get_product(int(identifier))
        if not image_usable(a):
            continue
        x = image_base(a, scales)
        mask = np.concatenate((a["image_valid"], a["image_valid"]))
        for k in range(2 * IMAGE_BANDS):
            values = x[k][mask[k]]
            count[k] += len(values)
            sums[k] += values.sum()
            squares[k] += np.dot(values, values)
    check(np.all(count > 0), "FTRAIN image moments supported")
    mean = sums / count
    variance = squares / count - mean**2
    check(np.all(variance >= -1e-10), "nonnegative image variance")
    std = np.sqrt(np.maximum(variance, 0))
    std[std <= 0] = 1
    return ImageNormalizer(
        scale_ujy=scales,
        scale_fallback_RMS=fallback,
        mean=mean.tolist(),
        std=std.tolist(),
        observed_pixels=count.tolist(),
    )


def fit_spectrum_floor(
    ids: Sequence[int] | NDArray[np.integer], get_product: Callable[[int], Product]
) -> float:
    buffer = np.empty(len(ids) * SPECTRUM_LENGTH, np.float32)
    used = 0
    for identifier in ids:
        a = get_product(int(identifier))
        if bool(a["spectrum_present"]):
            values = a["spectrum_error_cgs"][a["spectrum_valid"]]
            buffer[used : used + len(values)] = values
            used += len(values)
    check(used > 0, "nonempty SSL spectra for training-only floor")
    floor = 1e-6 * median_f32_inplace(buffer[:used])
    check(np.isfinite(floor) and floor > 0, "positive SSL spectral floor")
    return floor


def fit_normalizers(
    train_ids: Sequence[int] | NDArray[np.integer],
    ssl_ids: Sequence[int] | NDArray[np.integer],
    get_product: Callable[[int], Product],
    train_flux: NDArray[np.floating],
    train_error: NDArray[np.floating],
    train_valid: NDArray[np.bool_],
) -> Normalizers:
    image = fit_image(train_ids, get_product)
    table = fit_table(train_flux, train_error, train_valid)
    floor = fit_spectrum_floor(ssl_ids, get_product)
    return Normalizers(image=image, table=table, spectrum=SpectrumNormalizer(floor_cgs=floor))
