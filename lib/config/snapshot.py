"""Capture and restore the text configuration used by an experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

from lib.config import FlowConfig
from lib.config.experiment import (
    Experiment,
    External,
    ImageSplit,
    Lightning,
    RecipeMethod,
    image_splits,
)
from lib.utils.artifacts import sha256_file

_TEXT_CONFIG = {".yaml", ".yml", ".json", ".conf", ".toml"}
_active: ContextVar[Path | None] = ContextVar("config_snapshot", default=None)


def public_config(value: Any) -> Any:
    if isinstance(value, list):
        return [public_config(item) for item in value]
    if isinstance(value, dict):
        return {
            key: public_config(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") not in {"tracking_uri", "mlflow_tracking_uri"}
        }
    return value


def _config_bytes(source: Path) -> bytes:
    raw = source.read_bytes()
    if b"tracking_uri" not in raw.lower().replace(b"-", b"_"):
        return raw
    if source.suffix.lower() in {".yaml", ".yml", ".json"}:
        if source.suffix.lower() == ".json":
            return json.dumps(public_config(json.loads(raw)), indent=2).encode()
        return yaml.safe_dump(
            public_config(yaml.safe_load(raw)), sort_keys=False, allow_unicode=True
        ).encode()
    return b"".join(
        line
        for line in raw.splitlines(keepends=True)
        if b"tracking_uri" not in line.lower().replace(b"-", b"_")
    )


def _create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent) as stream:
        stream.write(data)
        stream.flush()
        try:
            os.link(stream.name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"Configuration changed at {path}") from None


def record_config(source: str | Path, snapshot: Path | None = None) -> Path | None:
    root = snapshot or _active.get()
    source = Path(source)
    if root is None or source.suffix.lower() not in _TEXT_CONFIG or not source.is_file():
        return None
    # Preserve the supplied name even when a source is a symlink.
    source = Path(os.path.abspath(source))
    if source.is_relative_to(root.resolve() / "resolved"):
        return source
    cwd = Path.cwd()
    if source.is_relative_to(cwd):
        target = root / "files/workspace" / source.relative_to(cwd)
    else:
        target = root / "files/absolute" / source.relative_to(source.anchor)
    try:
        content = _config_bytes(source)
    except (OSError, UnicodeError, yaml.YAMLError, json.JSONDecodeError):
        # The method reports its unreadable recipe. Omit bytes whose secrets cannot be removed.
        return None
    _create(target, content)
    return target


@contextmanager
def recording_configs(directory: Path | None) -> Iterator[None]:
    if directory is None:
        yield
        return
    token = _active.set(directory)
    try:
        yield
    finally:
        _active.reset(token)


def create_snapshot(directory: Path, config: Experiment | FlowConfig) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    # Preserve fields_set. For example, split_config owns the seed and train/test proportions.
    replay = public_config(config.model_dump(mode="json", exclude_unset=True))
    effective = public_config(config.model_dump(mode="json"))
    _create(
        directory / "flow.yaml",
        yaml.safe_dump(replay, sort_keys=False, allow_unicode=True).encode(),
    )
    _create(
        directory / "effective.json", json.dumps(effective, indent=2, ensure_ascii=False).encode()
    )
    files: list[str | Path | None] = []
    if isinstance(config, Experiment):
        if config.dataset:
            files.extend((config.dataset.feature_catalog, config.dataset.split_config))
        files.extend(
            method.recipe for method in config.methods.values() if isinstance(method, RecipeMethod)
        )
    else:
        if config.dataset:
            files.extend(config.dataset.get(key) for key in ("features", "data_config"))
        if config.train:
            files.append(config.train.get("config"))
        files.extend(m.get("config") for m in config.models)
        if config.euclid:
            e = config.euclid
            files.extend(
                s.config for s in (e.pretrain, e.finetune, e.classical, e.report) if s and s.enabled
            )
    for source in files:
        if source is not None:
            record_config(source, directory)
    seal_snapshot(directory)
    return directory


def resolve_methods(config: Experiment, snapshot: Path, data: dict[str, Any]) -> Experiment:
    """Expose inline parameters as run-owned files for the existing backend APIs."""
    methods = dict(config.methods)
    for name, method in methods.items():
        if not isinstance(method, RecipeMethod) or method.recipe is not None:
            continue
        path = snapshot / "resolved" / f"{name}.yaml"
        raw = (
            method.params
            if isinstance(method, External)
            else {
                "estimator": method.model,
                "estimator_params": method.params,
            }
        )
        update: dict[str, Any] = {"recipe": path, "params": {}}
        if not isinstance(method, External):
            update["model"] = None
        if config.dataset and config.dataset.id_column and not method.report.id_column:
            update["report"] = method.report.model_copy(
                update={"id_column": config.dataset.id_column}
            )
        if isinstance(method, Lightning):
            if not data:
                continue  # Dataset preparation failed. Its error is reported against this method.
            assert config.dataset is not None
            assert method.train is not None
            ds = config.dataset
            images = image_splits(ds, method.split)
            raw.update(
                model_type="lightning",
                device=method.device,
                data_dir=str(images["train"]),
                catalog=data["train"],
                image_catalog={
                    "id_column": data["id_column"],
                    "target_column": data["target"],
                    "features": data["features"] if method.use_features else [],
                    "array_key": ds.array_key,
                },
                out_size=ds.image_size,
                crop_size=ds.crop_size,
                train=method.train.model_dump(exclude_unset=True),
            )
            held = data["evaluation_files"] or {method.split: data["test"]}
            update["predict_on"] = {
                label: ImageSplit(images=images[label], catalog=catalog, role="held-out")
                for label, catalog in held.items()
            }
        _create(
            path, yaml.safe_dump(public_config(raw), sort_keys=False, allow_unicode=True).encode()
        )
        methods[name] = method.model_copy(update=update)
    return config.model_copy(update={"methods": methods})


def seal_snapshot(directory: Path) -> None:
    files = [directory / "effective.json", directory / "flow.yaml"]
    files += sorted((directory / "resolved").glob("*.yaml"))
    files += sorted(p for p in (directory / "files").rglob("*") if p.is_file())
    hashes = {
        p.relative_to(directory).as_posix(): sha256_file(p)
        for p in files
    }
    data = {"version": 1, "working_directory": str(Path.cwd()), "sha256": hashes}
    manifest = directory / "manifest.json"
    with NamedTemporaryFile(dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(data, indent=2, ensure_ascii=False).encode())
            stream.flush()
            os.replace(temporary, manifest)
        finally:
            temporary.unlink(missing_ok=True)


def restore_snapshot(
    directory: Path, workspace: Path, *, absolute_roots: tuple[Path, ...] = ()
) -> Path:
    """Restore configs in their original layout. Existing files must already match the snapshot."""
    workspace = workspace.resolve()
    allowed = tuple(p.resolve() for p in absolute_roots)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["version"] != 1:
        raise ValueError("Unsupported configuration snapshot version")
    if not {"flow.yaml", "effective.json"} <= manifest["sha256"].keys():
        raise ValueError("Snapshot manifest is missing the flow configuration")
    pending: list[tuple[Path, bytes]] = []
    for stored, expected in manifest["sha256"].items():
        key = Path(stored)
        if key.is_absolute() or ".." in key.parts:
            raise ValueError(f"Invalid snapshot path {key}")
        source = (directory / key).resolve()
        if not source.is_relative_to(directory.resolve()):
            raise ValueError(f"Invalid snapshot source {source}")
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Configuration digest mismatch at {key}")
        if stored in {"flow.yaml", "effective.json"} or key.parts[:1] == ("resolved",):
            continue
        if key.parts[:2] == ("files", "workspace"):
            target = (workspace / key.relative_to("files/workspace")).resolve()
            if not target.is_relative_to(workspace):
                raise ValueError(f"Destination outside workspace: {target}")
        elif key.parts[:2] == ("files", "absolute"):
            target = (Path("/") / key.relative_to("files/absolute")).resolve()
            if not any(target.is_relative_to(root) for root in (workspace, *allowed)):
                raise ValueError(
                    f"Restoring {target} requires --absolute-root for its original location"
                )
        else:
            raise ValueError(f"Unknown snapshot path {key}")
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(f"Different configuration already exists at {target}")
        pending.append((target, content))
    for path, content in pending:
        _create(path, content)
    return directory.resolve() / "flow.yaml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Restore configuration files from a run snapshot.")
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--absolute-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    print(restore_snapshot(args.snapshot, args.workspace, absolute_roots=tuple(args.absolute_root)))
