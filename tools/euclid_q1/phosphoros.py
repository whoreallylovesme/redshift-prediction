from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import click
import numpy as np
import pandas as pd
from astropy.io import fits
from numpy.typing import NDArray
from pandas.api.types import is_integer_dtype
from pydantic import BaseModel, Field

from lib.dataloaders.euclid_transforms import BANDS
from lib.utils.log import get_logger
from lib.utils.utils import check, load_yaml_as

logger = get_logger()

type Raw = dict[str, NDArray[Any]]

ANCILLARY = (
    "GAL_EBV",
    "GAL_EBV_ERR",
    "FLUX_DETECTION_TOTAL",
    "FLUXERR_DETECTION_TOTAL",
    "FLUX_NIR_STACK_2FWHM_APER",
    "FLUXERR_NIR_STACK_2FWHM_APER",
)
RAW_KEYS = (
    "object_id",
    "tile",
    "field_code",
    "split",
    "bands",
    "ancillary_columns",
    "flux",
    "error",
    "ancillary_values",
    "vis_det",
    "vis_det_missing",
)
ROLES = ("train", "valid", "test")
SPLITS = (*ROLES, "unused")
FIELDS = {0: "EDF-N", 1: "EDF-F", 2: "EDF-S"}
AXIS = np.arange(601, dtype=np.float64) / 100
ROUNDOFF = 1e-12
ERROR_POLICY = "conditional_ratio_error_approximation"
PENDING = "DETECTION_KNOWN_PENDING_PHOTOMETRY_ELIGIBILITY"
UNKNOWN_DETECTION = "UNSUPPORTED_NATIVE_VIS_DET"
ELIGIBLE = "ENGINE_ELIGIBLE"
INELIGIBLE = "UNSUPPORTED_BANDS_OR_DUST"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
MAPPING_HEADER = (
    "# Filter, Flux Column, Error Column, Upper Limit/error ratio, "
    "Convert from MAG, Filter Shift Column"
)
SED_DIR = "COSMOS_public_el_sigma200"
EUCLID_FILTERS = ("VIS.vis", "NISP.Y", "NISP.J", "NISP.H")
REDDENING_CURVES = ("SB_Calzetti.dat", "SB_Calzetti_mod.dat", "SMC_Prevot.dat", "Gordon23/G23.dat")
LUMINOSITY_CURVE = "LuminosityFunctionCurves/PublicQ1/Mr_flat_minus24_to0.dat"


@dataclass(frozen=True)
class Side:
    name: str
    field_codes: tuple[int, ...]
    tiles: tuple[int, int]
    bands: dict[str, str]
    beta: dict[str, float]
    zpc: dict[str, float]
    zp_policy: str
    filters: tuple[str, ...]


NORTH = Side(
    name="north",
    field_codes=(0,),
    tiles=(102157301, 102160877),
    bands={
        "u": "U_EXT_MEGACAM",
        "g": "G_EXT_HSC",
        "r": "R_EXT_MEGACAM",
        "i": "I_EXT_PANSTARRS",
        "z": "Z_EXT_HSC",
        "vis": "VIS",
        "y": "Y",
        "j": "J",
        "h": "H",
    },
    beta={
        "u": 0.05, "g": 0.02, "r": 0.02, "i": 0.02, "z": 0.02,
        "vis": 0.02, "y": 0.05, "j": 0.05, "h": 0.05,
    },
    zpc={
        "u": 1.0, "g": 1.0, "r": 0.99, "i": 1.09, "z": 1.0,
        "vis": 1.05, "y": 0.95, "j": 0.97, "h": 0.95,
    },
    zp_policy="paper_v1_table1_north",
    filters=(
        "CFHT/MegaCam/MegaCam.u",
        "Subaru/HSC/HSC.g",
        "CFHT/MegaCam/MegaCam.r",
        "PAN-STARRS/PS1.i",
        "Subaru/HSC/HSC.z",
        "Euclid/VIS.vis",
        "Euclid/NISP.Y",
        "Euclid/NISP.J",
        "Euclid/NISP.H",
    ),
)
SOUTH = Side(
    name="south",
    field_codes=(1, 2),
    tiles=(102018000, 102047999),
    bands={
        "g": "G_EXT_DECAM",
        "r": "R_EXT_DECAM",
        "i": "I_EXT_DECAM",
        "z": "Z_EXT_DECAM",
        "vis": "VIS",
        "y": "Y",
        "j": "J",
        "h": "H",
    },
    beta={
        "g": 0.02, "r": 0.02, "i": 0.02, "z": 0.02,
        "vis": 0.02, "y": 0.05, "j": 0.05, "h": 0.05,
    },
    zpc={
        "g": 1.0, "r": 1.0, "i": 1.0, "z": 1.015,
        "vis": 1.05, "y": 0.95, "j": 0.97, "h": 0.95,
    },
    zp_policy="paper_v1_table1_south",
    filters=(
        "CTIO/DECam/DECam.g",
        "CTIO/DECam/DECam.r",
        "CTIO/DECam/DECam.i",
        "CTIO/DECam/DECam.z",
        "Euclid/VIS.vis",
        "Euclid/NISP.Y",
        "Euclid/NISP.J",
        "Euclid/NISP.H",
    ),
)
SIDES = {side.name: side for side in (NORTH, SOUTH)}


class SideConfig(BaseModel):
    workspace: Path
    model_grid_config: Path
    galactic_grid_config: Path
    redshift_config: Path
    grid_export: Path


class SetConfig(BaseModel):
    side: Literal["north", "south"]
    photometry: Path
    field_code: int | None = None
    split: str | None = None
    ids_table: Path | None = None
    ids_column: str = "object_id"
    subset_column: str | None = None
    subset_value: str | None = None


class VariantConfig(BaseModel):
    sets: list[str]
    divide: dict[str, float] = Field(default_factory=dict)
    multiply: dict[str, float] = Field(default_factory=dict)
    multiply_json: Path | None = None
    multiply_json_keys: list[str] = Field(default_factory=list)


class SouthWorkspaceConfig(BaseModel):
    north_workspace: Path
    aux_archive: Path
    aux_archive_root: str = "AuxiliaryData_0.8"


class PhosphorosConfig(BaseModel):
    executables: Path
    threads: int = 6
    buffer_size: int = 32
    timeout: int = 7200
    output_root: Path
    sides: dict[str, SideConfig]
    sets: dict[str, SetConfig]
    variants: dict[str, VariantConfig] = Field(default_factory=dict)
    south_workspace: SouthWorkspaceConfig | None = None


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def load_photometry(path: Path) -> Raw:
    with np.load(path, allow_pickle=False) as packed:
        return {key: packed[key] for key in RAW_KEYS}


def select_rows(raw: Raw, cfg: SetConfig) -> NDArray[np.int64]:
    ids = raw["object_id"]
    if cfg.ids_table is None:
        rows = np.arange(len(ids), dtype=np.int64)
    else:
        columns = [cfg.ids_column] + ([cfg.subset_column] if cfg.subset_column else [])
        table = pd.read_parquet(cfg.ids_table, columns=columns)
        if cfg.subset_column is not None:
            table = table[table[cfg.subset_column] == cfg.subset_value]
        position = {int(v): i for i, v in enumerate(ids)}
        wanted = table[cfg.ids_column].to_numpy(np.int64)
        missing = [int(v) for v in wanted if int(v) not in position]
        check(not missing, f"{len(missing)} requested ids are absent from {cfg.photometry}")
        rows = np.array([position[int(v)] for v in wanted], dtype=np.int64)
    keep = np.ones(len(rows), dtype=np.bool_)
    if cfg.field_code is not None:
        keep &= raw["field_code"][rows] == cfg.field_code
    if cfg.split is not None:
        keep &= raw["split"][rows] == cfg.split
    return rows[keep]


def observations(
    raw: Raw, rows: NDArray[np.int64], side: Side
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = raw["object_id"]
    check(len(rows) and len(np.unique(rows)) == len(rows), "unique selected rows required")
    check(ids.dtype == np.int64 and len(np.unique(ids)) == len(ids), "unique int64 ids required")
    check(
        tuple(raw["bands"].tolist()) == BANDS
        and tuple(raw["ancillary_columns"].tolist()) == ANCILLARY,
        "native band or ancillary order differs",
    )
    check(
        raw["flux"].dtype == np.float64
        and raw["error"].dtype == np.float64
        and raw["ancillary_values"].dtype == np.float64,
        "float64 native photometry required",
    )
    field, split, tile = raw["field_code"][rows], raw["split"][rows], raw["tile"][rows]
    check(np.isin(field, side.field_codes).all(), f"field outside the {side.name} side")
    check(np.isin(split, ROLES).all(), "unknown split")
    low, high = side.tiles
    check(((tile >= low) & (tile <= high)).all(), f"owner tiles outside the {side.name} side")
    detection = raw["vis_det"][rows]
    known = ~raw["vis_det_missing"][rows] & np.isin(detection, [0, 1])
    ledger = pd.DataFrame(
        {
            "object_id": ids[rows],
            "tile": tile,
            "split": split,
            "field_code": field,
            "input_status": np.where(known, PENDING, UNKNOWN_DETECTION),
        }
    )
    chosen = rows[known]
    frame = pd.DataFrame(
        {
            "object_id": ids[chosen],
            "tile": raw["tile"][chosen],
            "split": raw["split"][chosen],
            "vis_det": raw["vis_det"][chosen],
        }
    )
    frame["object_id"] = pd.array(ids[chosen], dtype="Int64")
    lookup = {name: j for j, name in enumerate(BANDS)}
    for alias, name in side.bands.items():
        frame[f"flux_{alias}"] = raw["flux"][chosen, lookup[name]]
        frame[f"fluxerr_{alias}"] = raw["error"][chosen, lookup[name]]
    ancillary = raw["ancillary_values"][chosen]
    frame["gal_ebv"], frame["gal_ebv_err"] = ancillary[:, 0], ancillary[:, 1]
    frame["flux_detection_total"] = ancillary[:, 2]
    frame["fluxerr_detection_total"] = ancillary[:, 3]
    detected = frame["vis_det"].to_numpy() == 1
    frame["flux_detection_aper"] = np.where(detected, frame["flux_vis"], ancillary[:, 4])
    frame["fluxerr_detection_aper"] = np.where(detected, frame["fluxerr_vis"], ancillary[:, 5])
    return frame, ledger


def manifest(frame: pd.DataFrame, side: Side, include_unused: bool = False) -> pd.DataFrame:
    columns = ["object_id", "tile", "split"]
    check(set(columns) <= set(frame.columns), "manifest columns missing")
    for column in columns[:2]:
        values = frame[column]
        check(
            not values.isna().any() and is_integer_dtype(values.dtype),
            f"{column} must be non-null integers",
        )
    check(not frame["object_id"].duplicated().any(), "duplicate object_id")
    check(frame["split"].isin(SPLITS).all(), "unknown split")
    check(
        not (frame.groupby("tile")["split"].nunique() > 1).any(),
        "one owner tile occurs in several splits",
    )
    selected = frame.loc[:, columns].copy()
    if not include_unused:
        selected = selected[selected["split"] != "unused"]
    check(not selected.empty, "empty manifest")
    low, high = side.tiles
    check(selected["tile"].between(low, high).all(), f"owner tiles outside the {side.name} side")
    return selected.reset_index(drop=True)


def build_inputs(observed: pd.DataFrame, side: Side) -> pd.DataFrame:
    selected = manifest(observed, side)
    raw = observed.set_index("object_id").loc[selected["object_id"]].reset_index()
    check(raw["vis_det"].isin([0, 1]).all(), "VIS_DET must be 0 or 1")
    output = selected.copy()
    output["vis_det"] = raw["vis_det"].to_numpy()
    for column in ("gal_ebv", "gal_ebv_err"):
        output[column] = raw[column].to_numpy(dtype=np.float64)
    output["valid_gal_ebv"] = np.isfinite(output["gal_ebv"]) & (output["gal_ebv"] >= 0)
    total = raw["flux_detection_total"].to_numpy(dtype=np.float64)
    total_error = raw["fluxerr_detection_total"].to_numpy(dtype=np.float64)
    aperture = raw["flux_detection_aper"].to_numpy(dtype=np.float64)
    aperture_error = raw["fluxerr_detection_aper"].to_numpy(dtype=np.float64)
    valid_total = np.isfinite(total) & (total > 0) & np.isfinite(total_error) & (total_error > 0)
    valid_ratio = (
        valid_total
        & np.isfinite(aperture)
        & (aperture > 0)
        & np.isfinite(aperture_error)
        & (aperture_error > 0)
    )
    ratio = np.full(len(raw), np.nan, dtype=np.float64)
    np.divide(total, aperture, out=ratio, where=valid_ratio)
    valid_ratio &= np.isfinite(ratio) & (ratio > 0)
    masks = []
    for band, beta in side.beta.items():
        band_aperture = raw[f"flux_{band}"].to_numpy(dtype=np.float64)
        band_error = raw[f"fluxerr_{band}"].to_numpy(dtype=np.float64)
        direct = (
            (raw["vis_det"] == 1).to_numpy()
            if band == "vis"
            else np.zeros(len(raw), dtype=np.bool_)
        )
        flux = ratio * band_aperture
        error = np.abs(ratio) * band_error
        valid = (
            valid_ratio & np.isfinite(band_aperture) & np.isfinite(band_error) & (band_error > 0)
        )
        flux[direct] = total[direct]
        error[direct] = total_error[direct]
        valid[direct] = valid_total[direct]
        valid &= np.isfinite(flux) & np.isfinite(error) & (error > 0)
        flux[~valid] = np.nan
        error[~valid] = np.nan
        factor = side.zpc[band]
        flux *= factor
        error *= factor
        output[f"flux_{band}"] = flux
        output[f"fluxerr_{band}"] = np.hypot(error, beta * flux)
        masks.append(valid)
    output["n_valid_bands"] = np.stack(masks).sum(axis=0).astype(np.int16)
    return output


def write_fits(frame: pd.DataFrame, path: Path, side: Side) -> None:
    manifest(frame, side, include_unused=True)
    columns = [
        fits.Column(name="OBJECT_ID", format="K", array=frame["object_id"].to_numpy(np.int64))
    ]
    for band in side.bands:
        flux = frame[f"flux_{band}"].to_numpy(dtype=np.float64).copy()
        error = frame[f"fluxerr_{band}"].to_numpy(dtype=np.float64).copy()
        valid = np.isfinite(flux) & np.isfinite(error) & (error > 0)
        flux[~valid] = np.nan
        error[~valid] = np.nan
        columns.extend(
            [
                fits.Column(name=f"FLUX_{band.upper()}", format="D", unit="uJy", array=flux),
                fits.Column(name=f"FLUXERR_{band.upper()}", format="D", unit="uJy", array=error),
            ]
        )
    for column in ("gal_ebv", "gal_ebv_err"):
        columns.append(
            fits.Column(
                name=column.upper(),
                format="D",
                unit="mag",
                array=frame[column].to_numpy(dtype=np.float64),
            )
        )
    table = fits.BinTableHDU.from_columns(columns, name="PHOSPHOROS_PUBLIC_INPUTS")
    table.header["EXACTQ1"] = False
    table.header["FLXUNIT"] = "uJy"
    table.header["ERRPOL"] = ERROR_POLICY
    table.header["RATIOUNC"] = False
    table.header["COVAR"] = False
    table.header["ZPCPOL"] = side.zp_policy
    check(not path.exists(), f"refusing to overwrite {path}")
    fits.HDUList([fits.PrimaryHDU(), table]).writeto(path, overwrite=False)


def prepare_set(cfg: PhosphorosConfig, name: str) -> None:
    set_cfg = cfg.sets[name]
    side = SIDES[set_cfg.side]
    raw = load_photometry(set_cfg.photometry)
    rows = select_rows(raw, set_cfg)
    observed, ledger = observations(raw, rows, side)
    converted = build_inputs(observed, side)
    eligible = ((converted["n_valid_bands"] >= 2) & converted["valid_gal_ebv"]).to_numpy(bool)
    check(eligible.any(), f"{name}: no engine-eligible object")
    converted_ids = converted["object_id"].to_numpy(np.int64)
    status = pd.Series(np.where(eligible, ELIGIBLE, INELIGIBLE), index=converted_ids)
    known = ledger["object_id"].isin(converted_ids).to_numpy()
    ledger.loc[known, "input_status"] = status.reindex(ledger["object_id"][known]).to_numpy()
    bands = pd.Series(converted["n_valid_bands"].to_numpy(), index=converted_ids)
    ledger["n_valid_bands"] = bands.reindex(ledger["object_id"].to_numpy()).astype("Int16").array
    out = cfg.output_root / name / "preparation"
    out.mkdir(parents=True, exist_ok=False)
    write_fits(converted.loc[eligible].copy(), out / "computational_inputs.fits", side)
    ledger.to_parquet(out / "ledger.parquet", index=False)
    counts = ledger["input_status"].value_counts().to_dict()
    logger.info("%s %s: requested %d eligible %d %s", side.name, name, len(ledger),
                int(eligible.sum()), counts)


def set_dir(cfg: PhosphorosConfig, name: str, variant: str | None) -> Path:
    return cfg.output_root / name if variant is None else cfg.output_root / variant / name


def predictions_dir(cfg: PhosphorosConfig, variant: str | None) -> Path:
    root = cfg.output_root if variant is None else cfg.output_root / variant
    return root / "predictions"


def variant_factors(variant: VariantConfig) -> tuple[dict[str, float], dict[str, float]]:
    multiply = dict(variant.multiply)
    if variant.multiply_json is not None:
        node: Any = json.loads(variant.multiply_json.read_text())
        for key in variant.multiply_json_keys:
            node = node[key]
        multiply = {str(band): float(factor) for band, factor in node.items()}
    return dict(variant.divide), multiply


def write_variant(
    source: Path, target: Path, divide: dict[str, float], multiply: dict[str, float]
) -> None:
    with fits.open(source, memmap=False) as hdus:
        table = hdus[1].copy()
        for band in dict.fromkeys([*divide, *multiply]):
            for column in (f"FLUX_{band}", f"FLUXERR_{band}"):
                value = np.asarray(table.data[column], np.float64)
                if band in divide:
                    value = value / divide[band]
                if band in multiply:
                    value = value * multiply[band]
                table.data[column] = value
        fits.HDUList([hdus[0].copy(), table]).writeto(target)


def binary(cfg: PhosphorosConfig, name: str) -> str:
    return str((cfg.executables / name).absolute())


def engine_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({key: str(threads) for key in THREAD_VARIABLES})
    env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def call(command: list[str], log: Path, cwd: Path, threads: int, timeout: int) -> None:
    with log.open("x") as stream:
        done = subprocess.run(
            command,
            env=engine_env(threads),
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
            cwd=str(cwd),
        )
    check(done.returncode == 0, f"{command[0]} exited with {done.returncode}, see {log}")


def run_set(cfg: PhosphorosConfig, name: str, variant: str | None) -> None:
    side = cfg.sides[cfg.sets[name].side]
    catalog = (cfg.output_root / name / "preparation" / "computational_inputs.fits").resolve()
    target = set_dir(cfg, name, variant).resolve()
    if variant is not None:
        target.mkdir(parents=True, exist_ok=False)
        divide, multiply = variant_factors(cfg.variants[variant])
        write_variant(catalog, target / "computational_inputs.fits", divide, multiply)
        catalog = target / "computational_inputs.fits"
    execution = target / "execution"
    execution.mkdir(exist_ok=False)
    command = [
        binary(cfg, "PhosphorosComputeRedshifts"),
        f"--config-file={side.redshift_config.resolve()}",
        f"--phosphoros-root={side.workspace.resolve()}",
        f"--input-catalog-file={catalog}",
        f"--phz-output-dir={execution / 'results'}",
        f"--thread-no={cfg.threads}",
        f"--input-buffer-size={cfg.buffer_size}",
        f"--log-file={execution / 'engine.log'}",
    ]
    logger.info("run %s%s", name, f" ({variant})" if variant else "")
    call(command, execution / "stdout.log", execution, cfg.threads, cfg.timeout)


def pdf_status(pdf: NDArray[np.float64], cdf: NDArray[np.float64]) -> NDArray[np.int16]:
    finite = np.asarray(np.isfinite(pdf).all(axis=1))
    peak = np.max(np.where(finite[:, None], pdf, 0.0), axis=1)
    massive = finite & (peak > 0)
    negative = massive & (pdf < -ROUNDOFF * peak[:, None]).any(axis=1)
    total = cdf[:, -1]
    status = np.zeros(len(pdf), dtype=np.int16)
    status[~finite] = 1
    status[finite & ~massive] = 3
    status[negative] = 2
    status[massive & ~negative & ~(np.isfinite(total) & (total > 0))] = 3
    return status


def quantile_index(cdf: NDArray[np.float64], q: float) -> NDArray[np.intp]:
    index: NDArray[np.intp] = np.argmax(cdf >= q * cdf[:, -1:], axis=1)
    return index


def posterior(phz: Path, ids: NDArray[np.int64]) -> pd.DataFrame:
    with fits.open(phz, memmap=False) as hdus:
        table = hdus[1]
        comments = "".join(
            str(card.value) for card in table.header.cards if card.keyword == "COMMENT"
        )
        declared = re.findall(r"Z-BINS\s*:\s*\{([^}]+)\}", comments)
        check(len(declared) == 1, "expected one Z-BINS declaration in the Core output")
        axis = np.asarray([float(v.strip()) for v in declared[0].split(",")], dtype=np.float64)
        check(
            axis.shape == AXIS.shape and np.allclose(axis, AXIS, rtol=0, atol=1e-9),
            "Core posterior knots differ from the 0..6 step 0.01 grid",
        )
        check(table.columns["ID"].format == "K", "Core ID must be int64")
        data = table.data
        oid = np.asarray(data["ID"], dtype=np.int64)
        pdf = np.asarray(data["Z-1D-PDF"], dtype=np.float64)
        peak_z = np.asarray(data["1DPDF-Peak-Z"], dtype=np.float64)
        best_z = np.asarray(data["Z"], dtype=np.float64)
        missing = np.asarray(data["Missing-data-flag"], dtype=np.bool_)
    check(np.array_equal(oid, ids), "Core output order differs from the input catalogue")
    cdf = np.cumsum(np.maximum(pdf, 0), axis=1, dtype=np.float64)
    status = pdf_status(pdf, cdf)
    valid = status == 0
    good = cdf[valid]
    median = np.full(len(oid), np.nan)
    lo = np.full(len(oid), np.nan)
    hi = np.full(len(oid), np.nan)
    median[valid] = AXIS[quantile_index(good, 0.5)]
    lo[valid] = AXIS[quantile_index(good, 0.16)]
    hi[valid] = AXIS[quantile_index(good, 0.84)]
    return pd.DataFrame(
        {
            "object_id": oid,
            "ppub_median": median,
            "pdf_status": status,
            "engine_missing_photometry_flag": missing,
            "ppub_lo68": lo,
            "ppub_hi68": hi,
            "ppub_mode": np.where(valid, peak_z, np.nan),
            "ppub_bestfit_z": np.where(valid, best_z, np.nan),
        }
    )


def assemble(
    ledger: pd.DataFrame, points: pd.DataFrame, name: str, config_id: str
) -> pd.DataFrame:
    eligible = ledger["input_status"] == ELIGIBLE
    check(
        set(ledger["object_id"][eligible]) == set(points["object_id"]),
        "eligible set differs from the engine set",
    )
    aligned = ledger.merge(
        points, on="object_id", how="left", validate="one_to_one", indicator=True
    )
    present = aligned.pop("_merge") == "both"
    median = aligned["ppub_median"].to_numpy(np.float64)
    return pd.DataFrame(
        {
            "object_id": aligned["object_id"].to_numpy(np.int64),
            "target_set": name,
            "field": np.array([FIELDS[int(code)] for code in aligned["field_code"]]),
            "split": aligned["split"].to_numpy(),
            "config_id": config_id,
            "ppub_median": median,
            "ppub_mode": aligned["ppub_mode"].to_numpy(np.float64),
            "ppub_lo68": aligned["ppub_lo68"].to_numpy(np.float64),
            "ppub_hi68": aligned["ppub_hi68"].to_numpy(np.float64),
            "ppub_bestfit_z": aligned["ppub_bestfit_z"].to_numpy(np.float64),
            "input_status": aligned["input_status"].to_numpy(),
            "n_valid_bands": aligned["n_valid_bands"].astype("Int16"),
            "engine_present": present.to_numpy(bool),
            "pdf_status": aligned["pdf_status"].astype("Int16"),
            "engine_missing_photometry_flag": aligned["engine_missing_photometry_flag"].astype(
                "boolean"
            ),
            "has_prediction": np.isfinite(median),
        }
    )


def postprocess_set(cfg: PhosphorosConfig, name: str, variant: str | None) -> None:
    set_cfg = cfg.sets[name]
    target = set_dir(cfg, name, variant)
    source = target if variant is not None else target / "preparation"
    catalog = source / "computational_inputs.fits"
    with fits.open(catalog, memmap=False) as hdus:
        ids = np.asarray(hdus[1].data["OBJECT_ID"], dtype=np.int64)
    points = posterior(target / "execution" / "results" / "phz_cat.fits", ids)
    ledger = pd.read_parquet(cfg.output_root / name / "preparation" / "ledger.parquet")
    config_id = f"{set_cfg.side}_{sha256(cfg.sides[set_cfg.side].redshift_config)[:8]}"
    final = assemble(ledger, points, name, config_id)
    out = predictions_dir(cfg, variant)
    out.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out / f"{name}.parquet", index=False)
    logger.info("%s: requested %d engine %d with prediction %d -> %s", name, len(final),
                len(points), int(final["has_prediction"].sum()), out / f"{name}.parquet")


def copy_file(source: Path, target: Path) -> None:
    check(not target.exists(), f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def build_south_workspace(cfg: PhosphorosConfig) -> None:
    spec = cfg.south_workspace
    if spec is None:
        raise ValueError("south_workspace section required")
    root = cfg.sides["south"].workspace
    root.mkdir(parents=True, exist_ok=False)
    source = spec.north_workspace / "AuxiliaryData"
    aux = root / "AuxiliaryData"
    with tarfile.open(spec.aux_archive, "r:xz") as archive:
        for band in "griz":
            relative = f"Filters/CTIO/DECam/DECam.{band}.dat"
            member = archive.extractfile(f"{spec.aux_archive_root}/{relative}")
            if member is None:
                raise ValueError(f"{relative} missing from {spec.aux_archive}")
            (aux / relative).parent.mkdir(parents=True, exist_ok=True)
            (aux / relative).write_bytes(member.read())
    for name in EUCLID_FILTERS:
        copy_file(source / f"Filters/Euclid/{name}.dat", aux / f"Filters/Euclid/{name}.dat")
    for path in sorted((source / "SEDs" / SED_DIR).iterdir()):
        if path.is_file():
            copy_file(path, aux / "SEDs" / SED_DIR / path.name)
    copy_file(source / "SEDs/solar_spectrum.sed", aux / "SEDs/solar_spectrum.sed")
    for relative in REDDENING_CURVES:
        copy_file(source / "ReddeningCurves" / relative, aux / "ReddeningCurves" / relative)
    copy_file(source / LUMINOSITY_CURVE, aux / LUMINOSITY_CURVE)
    (aux / "AxisPriors").mkdir()
    catalog_type = config_values(cfg.sides["south"].model_grid_config)["catalog-type"]
    products = root / "IntermediateProducts" / catalog_type
    products.mkdir(parents=True)
    lines = [MAPPING_HEADER] + [
        f"{name} FLUX_{alias.upper()} FLUXERR_{alias.upper()} 3 0 NONE"
        for name, alias in zip(SOUTH.filters, SOUTH.bands, strict=True)
    ]
    (products / "filter_mapping.txt").write_text("\n".join(lines) + "\n")
    logger.info("south workspace ready at %s", root)


def build_grids(cfg: PhosphorosConfig, side_name: str) -> None:
    side = cfg.sides[side_name]
    workspace = side.workspace.resolve()
    model = config_values(side.model_grid_config)
    galactic = config_values(side.galactic_grid_config)
    products = workspace / "IntermediateProducts" / model["catalog-type"]
    grid = products / "ModelGrids" / model["output-model-grid"]
    check(not grid.exists(), f"refusing to overwrite {grid}")
    logs = (cfg.output_root / "grids" / side_name).resolve()
    logs.mkdir(parents=True, exist_ok=False)
    side.grid_export.parent.mkdir(parents=True, exist_ok=True)
    root = f"--phosphoros-root={workspace}"
    threads = f"--thread-no={cfg.threads}"
    steps = {
        "model_grid": [
            binary(cfg, "PhosphorosComputeModelGrid"),
            f"--config-file={side.model_grid_config.resolve()}",
            root,
            threads,
            f"--log-file={logs / 'model_grid.log'}",
        ],
        "galactic_grid": [
            binary(cfg, "PhosphorosComputeGalacticAbsorptionCoefficientGrid"),
            f"--config-file={side.galactic_grid_config.resolve()}",
            root,
            threads,
            f"--log-file={logs / 'galactic_grid.log'}",
        ],
        "export": [
            binary(cfg, "PhzModelGrid2Fits"),
            root,
            f"--catalog-type={galactic['catalog-type']}",
            f"--model-grid-file={galactic['model-grid-file']}",
            f"--output-file={side.grid_export.resolve()}",
            "--squash",
            f"--log-file={logs / 'export.log'}",
        ],
    }
    for step, command in steps.items():
        logger.info("%s grids: %s", side_name, step)
        call(command, logs / f"{step}.stdout.log", logs, cfg.threads, cfg.timeout)


def chosen_sets(cfg: PhosphorosConfig, names: tuple[str, ...], variant: str | None) -> list[str]:
    check(variant is None or variant in cfg.variants, f"unknown variant {variant}")
    available = list(cfg.sets) if variant is None else cfg.variants[variant].sets
    picked = list(names) or available
    unknown = sorted(set(picked) - set(available))
    check(not unknown, f"unknown sets {unknown}")
    return picked


def load_config(path: Path) -> PhosphorosConfig:
    return load_yaml_as(str(path), PhosphorosConfig)


CONFIG = click.option(
    "--config", "config_path", type=click.Path(exists=True, path_type=Path), required=True
)
SETS = click.option("--set", "names", multiple=True)
VARIANT = click.option("--variant", default=None)


@click.group()
def cli() -> None:
    pass


@cli.command("workspace")
@CONFIG
def workspace_command(config_path: Path) -> None:
    build_south_workspace(load_config(config_path))


@cli.command("grids")
@CONFIG
@click.option("--side", "side_name", type=click.Choice(sorted(SIDES)), required=True)
def grids_command(config_path: Path, side_name: str) -> None:
    build_grids(load_config(config_path), side_name)


@cli.command("prepare")
@CONFIG
@SETS
def prepare_command(config_path: Path, names: tuple[str, ...]) -> None:
    cfg = load_config(config_path)
    for name in chosen_sets(cfg, names, None):
        prepare_set(cfg, name)


@cli.command("run")
@CONFIG
@SETS
@VARIANT
def run_command(config_path: Path, names: tuple[str, ...], variant: str | None) -> None:
    cfg = load_config(config_path)
    for name in chosen_sets(cfg, names, variant):
        run_set(cfg, name, variant)


@cli.command("postprocess")
@CONFIG
@SETS
@VARIANT
def postprocess_command(config_path: Path, names: tuple[str, ...], variant: str | None) -> None:
    cfg = load_config(config_path)
    for name in chosen_sets(cfg, names, variant):
        postprocess_set(cfg, name, variant)


if __name__ == "__main__":
    cli()
