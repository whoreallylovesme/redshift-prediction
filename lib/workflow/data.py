from __future__ import annotations

import hashlib
import importlib
import json
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from lib.config import DatasetConfig, FeaturesConfig
from lib.config.experiment import Dataset
from lib.dataloaders.dataset import exact_index, matching_id_types
from lib.utils.utils import load_yaml_as


def symbol(spec: str) -> Any:
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError(f"Expected module:symbol, got {spec!r}")
    return getattr(importlib.import_module(module), name)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def code_version(plugins: list[str]) -> str:
    root = Path(__file__).resolve().parents[2]
    packages = {"lib", "runtime"}
    packages.update(p.split(":", 1)[0].split(".", 1)[0] for p in plugins if ":" in p)
    sha = hashlib.sha256()
    for package in sorted(packages):
        spec = find_spec(package)
        if spec is None:
            raise ValueError(f"Unknown plugin package {package}")
        for folder in spec.submodule_search_locations or []:
            base = Path(folder)
            for source in sorted(base.rglob("*.py")):
                sha.update(f"{package}/{source.relative_to(base)}".encode())
                sha.update(source.read_bytes())
        if not spec.submodule_search_locations and spec.origin:
            sha.update(package.encode())
            sha.update(Path(spec.origin).read_bytes())
    if (root / "uv.lock").exists():
        sha.update((root / "uv.lock").read_bytes())
    for dependency in ("scikit-learn", "numpy", "pandas", "pyarrow", "prefect"):
        sha.update(f"{dependency}={version(dependency)}".encode())
    return sha.hexdigest()


def table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    return pd.read_parquet(path)


def read_source(cfg: Dataset, path: Path, source_sha: str) -> pd.DataFrame:
    if cfg.reader:
        frame: pd.DataFrame = symbol(cfg.reader)(path)
    elif path.suffix.lower() == ".csv" and cfg.id_column:
        frame = pd.read_csv(path, dtype={cfg.id_column: cfg.csv_id_type}, low_memory=False)
    else:
        frame = table(path)
    if cfg.id_column is None:
        if "__source_row" in frame.columns:
            raise ValueError("Reserved generated row identifier already exists: __source_row")
        frame["__source_row"] = [f"{source_sha}:{i}" for i in range(len(frame))]
    return frame


def prepare_dataset(cfg: Dataset, out: Path) -> dict[str, Any]:
    input_sha = digest(cfg.source)
    frame = read_source(cfg, cfg.source, input_sha)
    if cfg.feature_catalog:
        catalog = load_yaml_as(str(cfg.feature_catalog), FeaturesConfig)
        target, features = catalog.target, catalog.feature_sets[str(cfg.feature_set)]
    else:
        target, features = str(cfg.target), cfg.features
    if len(set(features)) != len(features) or {
        target,
        cfg.id_column or "__source_row",
        cfg.split_column,
        cfg.weight_column,
    } & set(features):
        raise ValueError("Features must be unique and exclude target, identifier, weight and split")
    id_column = cfg.id_column or "__source_row"
    if target == id_column:
        raise ValueError("Dataset identifier and target must be distinct")
    source_ids = exact_index(frame[id_column], id_column)
    if cfg.cleaner:
        frame = symbol(cfg.cleaner)(frame, target)
        cleaned_ids = exact_index(frame[id_column], id_column)
        if not cleaned_ids.isin(source_ids).all():
            raise ValueError("Dataset cleaner introduced identifiers outside the source")
    else:
        frame = frame.dropna(subset=[target])
    if cfg.weight_column:
        weights = pd.to_numeric(frame[cfg.weight_column], errors="raise")
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Sample weights must be positive and finite")
    separate: dict[str, pd.DataFrame] = {}
    if cfg.evaluation_files:
        train = frame
        for label, path in cfg.evaluation_files.items():
            sha = digest(path)
            if sha == input_sha:
                raise ValueError(f"Evaluation split {label} repeats the training source")
            held = read_source(cfg, path, sha)
            identities = exact_index(held[id_column], id_column)
            if not matching_id_types(source_ids,identities):
                raise ValueError(
                    f"Evaluation split {label} has a different ID type; "
                    "use the same exact integer or string IDs in every source (csv_id_type)"
                )
            if len(source_ids.intersection(identities)):
                raise ValueError(f"Evaluation split {label} shares identifiers with training")
            separate[label] = held
        test = None
    elif cfg.split_column:
        train = frame[frame[cfg.split_column] == cfg.train_label]
        test = frame[frame[cfg.split_column] == cfg.test_label]
    else:
        if cfg.split_config:
            split = load_yaml_as(str(cfg.split_config), DatasetConfig)
            size, seed, shuffle = split.test_size, split.random_seed, split.shuffle
        else:
            size, seed, shuffle = cfg.test_size, cfg.seed, cfg.shuffle
        train, test = train_test_split(frame, test_size=size, random_state=seed, shuffle=shuffle)
    if train.empty or (test is not None and test.empty):
        raise ValueError("Both train and evaluation splits need observations")
    eligible = frame.select_dtypes(include=["number", "bool"]) if cfg.feature_catalog else frame
    missing = [name for name in features if name not in eligible]
    if missing and not cfg.feature_catalog:
        raise ValueError(f"Missing configured features: {missing}")
    features = [name for name in features if name in eligible]
    if not features:
        raise ValueError("None of the configured features are available")
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    files = {}
    for label, held in (separate or {"test": test}).items():
        if held is None or held.empty:
            raise ValueError(f"Empty evaluation split {label}")
        if absent := set(features).difference(held.columns):
            raise ValueError(f"Evaluation split {label} lacks features {sorted(absent)}")
        destination = out / (f"held_{label}.parquet" if separate else "test.parquet")
        held.to_parquet(destination, index=False)
        files[label] = str(destination)
    return {
        "train": str(out / "train.parquet"),
        "evaluation_files": files if separate else {},
        "test": files.get("test") if not separate else None,
        "features": features,
        "missing_features": missing,
        "feature_set": cfg.feature_set,
        "target": target,
        "id_column": id_column,
        "weight_column": cfg.weight_column,
        "train_rows": len(train),
        "test_rows": len(test) if test is not None else sum(map(len, separate.values())),
        "train_sha256": digest(out / "train.parquet"),
        "source_sha256": input_sha,
    }


def signature(*values: Any) -> str:
    raw = json.dumps(values, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()
