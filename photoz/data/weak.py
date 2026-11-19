from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from lib.config.snapshot import record_config
from lib.utils.utils import load_yaml_as
from photoz.data.identifiers import exact_ids as exact_ids

COHORT_COLUMNS = ["object_id", "tile", "split", "z_train", "label_source", "sample_weight"]
WEIGHTS = {"spe_halpha_quality": 1.0, "phz_quality": 0.3}


class TargetProtocol(BaseModel):
    cohort: Literal["weak", "large"]
    protocol: Path
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation: Path
    exclusions: Path
    identity_exclusions: Path | None = None


class PreparedCohort(TargetProtocol):
    observed: Path
    provenance: Path
    provenance_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest: Path


def validate_cohort(frame: pd.DataFrame, excluded: pd.DataFrame) -> None:
    if set(frame.split) != {"train", "valid"} or (frame.groupby("tile").split.nunique() > 1).any():
        raise ValueError("Expected nonempty, disjoint train/valid ownership")
    targets = frame.z_train.to_numpy(dtype=float)
    if (
        not np.isfinite(targets).all()
        or (targets <= 0).any()
        or not np.array_equal(
            frame.sample_weight.to_numpy(dtype=float),
            frame.label_source.map(WEIGHTS).to_numpy(dtype=float),
        )
    ):
        raise ValueError("Invalid weak redshift target or source weight")
    if (
        excluded.empty
        or frame.object_id.isin(excluded.object_id).any()
        or frame.tile.isin(excluded.tile).any()
    ):
        raise ValueError("Missing or overlapping independent test exclusions")


def checked_read(path: Path, sha: str, *, parquet: bool, columns: list[str] | None = None) -> Any:
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != sha:
            raise ValueError(f"Frozen digest mismatch: {path}")
        stream.seek(0)
        return pd.read_parquet(stream, columns=columns) if parquet else json.load(stream)


def load_prepared(recipe_path: Path) -> pd.DataFrame:
    cfg = load_yaml_as(str(recipe_path), PreparedCohort)
    root = recipe_path.parent
    receipt = checked_read(root / cfg.provenance, cfg.provenance_sha256, parquet=False)
    record_config(root / cfg.protocol)
    protocol = checked_read(root / cfg.protocol, cfg.protocol_sha256, parquet=False)
    if (
        receipt.get("desi_used_as_training_target") is not False
        or receipt.get("phz_unif_used") is not False
        or receipt.get("output_flux_unit") != "uJy"
        or receipt["sources_and_protocol"]["protocol"]["sha256"] != cfg.protocol_sha256
    ):
        raise ValueError("Prepared photometry provenance does not match the frozen raw-MER recipe")
    frame: pd.DataFrame = checked_read(
        root / cfg.observed, receipt["output"]["sha256"], parquet=True
    )
    if any("desi" in name.lower() or name.lower() == "targetid" for name in frame):
        raise ValueError("DESI reference columns in weak fitting data")
    frozen = checked_read(root / cfg.manifest, receipt["manifest"]["sha256"], parquet=True)
    for table in (frame, frozen):
        exact_ids(table)
        if table.object_id.duplicated().any():
            raise ValueError("Duplicate object in the prepared cohort")
    a = frame[COHORT_COLUMNS].sort_values("object_id").reset_index(drop=True)
    b = frozen[COHORT_COLUMNS].sort_values("object_id").reset_index(drop=True)
    if any(not np.array_equal(a[c].to_numpy(), b[c].to_numpy()) for c in COHORT_COLUMNS):
        raise ValueError("Observed IDs, ownership, targets or weights differ from frozen manifest")
    outputs = protocol["outputs"]
    excluded_hash = outputs["test_exclusions"]["sha256"]
    if receipt["exclusions_sha256"] != excluded_hash:
        raise ValueError("Observed cohort belongs to a different test exclusion manifest")
    excluded = checked_read(root / cfg.exclusions, excluded_hash, parquet=True)
    exact_ids(excluded)
    validate_cohort(a, excluded)
    valid = checked_read(
        root / cfg.validation, outputs["valid_memberships"]["sha256"], parquet=True
    )
    exact_ids(valid)
    if valid.object_id.duplicated().any():
        raise ValueError("Ambiguous validation ownership")
    matched = a.object_id.map(valid.set_index("object_id").tile.astype("Int64"))
    if (
        (a.split.eq("valid") != matched.notna()).any()
        or (matched.notna() & a.tile.ne(matched)).any()
        or (a.split.eq("train") & a.tile.isin(valid.tile)).any()
    ):
        raise ValueError("Prepared split differs from frozen validation membership")
    if cfg.cohort == "large" or "identity_exclusions_sha256" in receipt:
        guard = outputs["fitting_excluded_ids"]["sha256"]
        if cfg.identity_exclusions is None or receipt.get("identity_exclusions_sha256") != guard:
            raise ValueError("Large cohort requires a bound fitting identity embargo")
        identities = checked_read(root / cfg.identity_exclusions, guard, parquet=True)
        exact_ids(identities, ("object_id",))
        if (
            identities.empty
            or identities.object_id.duplicated().any()
            or a.object_id.isin(identities.object_id).any()
        ):
            raise ValueError("Invalid or overlapping fitting identity embargo")
    return frame
