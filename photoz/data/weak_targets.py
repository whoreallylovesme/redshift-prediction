"""Assign Euclid weak labels from quality-selected teacher tables and frozen spatial splits."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import pandas as pd
from pydantic import BaseModel

from lib.utils.artifacts import PinnedFile
from lib.utils.utils import load_yaml_as
from photoz.data.identifiers import exact_ids
from photoz.data.weak import WEIGHTS, TargetProtocol, checked_read, validate_cohort

CandidateTable = PinnedFile


class TargetSources(BaseModel):
    population: PinnedFile
    halpha: PinnedFile
    ambiguous_spe: PinnedFile


def assign_weak_targets(
    population: pd.DataFrame,
    halpha: pd.DataFrame,
    exclusions: pd.DataFrame,
    validation: pd.DataFrame,
    ambiguous_spe: pd.DataFrame,
    *,
    cohort: Literal["weak", "large"],
    identity_exclusions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one target per native ID. Hα takes priority, including SPE-only objects."""

    def native_keys(source: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
        exact_ids(source, keys)
        return source.astype(dict.fromkeys(keys, "int64"))

    source_keys = ("object_id", "tile")
    population = native_keys(population, source_keys)
    halpha = native_keys(halpha, source_keys)
    validation = native_keys(validation, source_keys)
    exclusions = native_keys(exclusions, source_keys)
    ambiguous_spe = native_keys(ambiguous_spe, ("object_id",))
    for source in (population, halpha, validation):
        if source.object_id.duplicated().any():
            raise ValueError(
                "Ambiguous object identity in candidate population or validation owners"
            )
    if cohort not in {"weak", "large"}:
        raise ValueError(f"Unknown cohort {cohort!r}")
    if cohort == "large" and (identity_exclusions is None or identity_exclusions.empty):
        raise ValueError("Large cohort requires the frozen fitting identity embargo")
    if identity_exclusions is not None:
        blocked = native_keys(identity_exclusions, ("object_id",)).object_id
        population = population.loc[~population.object_id.isin(blocked)]
        halpha = halpha.loc[~halpha.object_id.isin(blocked)]

    spe = halpha.loc[
        ~halpha.object_id.isin(ambiguous_spe.object_id), ["object_id", "tile", "spe_z"]
    ]
    spe = spe.rename(columns={"spe_z": "z_train"}).assign(
        label_source="spe_halpha_quality", sample_weight=WEIGHTS["spe_halpha_quality"]
    )
    phz = population.loc[
        ~population.object_id.isin(spe.object_id), ["object_id", "tile", "phz_median"]
    ]
    phz = phz.rename(columns={"phz_median": "z_train"}).assign(
        label_source="phz_quality", sample_weight=WEIGHTS["phz_quality"]
    )
    targets = pd.concat([phz, spe], ignore_index=True)
    held = targets.object_id.isin(exclusions.object_id) | targets.tile.isin(exclusions.tile)
    targets = targets.loc[~held].copy()
    owner = validation.set_index("object_id").tile
    valid = targets.object_id.isin(owner.index)
    targets["split"] = "train"
    targets.loc[valid, "split"] = "valid"
    # Map only the matched subset so missing IDs cannot coerce a 64-bit owner to float.
    targets.loc[valid, "tile"] = targets.loc[valid, "object_id"].map(owner)
    boundary = targets.split.eq("train") & targets.tile.isin(validation.tile)
    targets = targets.loc[~boundary].sort_values("object_id").reset_index(drop=True)
    validate_cohort(targets, exclusions)
    return targets


def prepare_targets(recipe: Path, sources: Path, output: Path) -> Path:
    """Materialize only the fitting manifest, from digest-bound target and geometry projections."""
    cfg = load_yaml_as(str(recipe), TargetProtocol)
    inputs = load_yaml_as(str(sources), TargetSources)
    frozen = checked_read(recipe.parent / cfg.protocol, cfg.protocol_sha256, parquet=False)

    def read_candidate(item: PinnedFile, columns: list[str]) -> pd.DataFrame:
        return checked_read(sources.parent / item.path, item.sha256, parquet=True, columns=columns)

    def read_protocol(key: str, path: Path, columns: list[str]) -> pd.DataFrame:
        sha = frozen["outputs"][key]["sha256"]
        return checked_read(recipe.parent / path, sha, parquet=True, columns=columns)

    identity_exclusions = None
    if cfg.cohort == "large":
        if (
            cfg.identity_exclusions is None
            or frozen.get("desi_used_for_training_or_selection") is not False
        ):
            raise ValueError("Large cohort requires a test-only DESI protocol and identity embargo")
        identity_exclusions = read_protocol(
            "fitting_excluded_ids", cfg.identity_exclusions, ["object_id"]
        )
        known = read_protocol(
            "all_desi_identity_exclusions",
            cfg.protocol.with_name("all_desi_identity_exclusions.parquet"),
            ["object_id"],
        )
        exact_ids(known, ("object_id",))
        if not known.object_id.isin(identity_exclusions.object_id).all():
            raise ValueError("A known DESI identity is missing from the fitting embargo")
    labels = assign_weak_targets(
        read_candidate(inputs.population, ["object_id", "tile", "phz_median"]),
        read_candidate(inputs.halpha, ["object_id", "tile", "spe_z"]),
        read_protocol("test_exclusions", cfg.exclusions, ["object_id", "tile"]),
        read_protocol("valid_memberships", cfg.validation, ["object_id", "tile"]),
        read_candidate(inputs.ambiguous_spe, ["object_id"]),
        cohort=cfg.cohort,
        identity_exclusions=identity_exclusions,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent) as tmp:
        labels.to_parquet(tmp, index=False)
        tmp.flush()
        os.link(tmp.name, output)
    return output
