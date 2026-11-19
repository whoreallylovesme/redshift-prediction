import tomllib
from importlib.metadata import EntryPoint, EntryPoints, entry_points
from pathlib import Path
from typing import Any

from lib.dataloaders.hdf5_rows import HDF5Rows as HDF5Rows


def _legacy_aliases() -> EntryPoints:
    group = f"astro.legacy.{__name__}"
    installed = entry_points(group=group)
    if installed:
        return installed
    manifest = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if manifest.is_file():
        with manifest.open("rb") as stream:
            declarations = tomllib.load(stream).get("project", {}).get("entry-points", {})
        if group in declarations:
            return EntryPoints(
                EntryPoint(name, target, group) for name, target in declarations[group].items()
            )
    return installed


_aliases = _legacy_aliases()
__all__ = ["HDF5Rows", *sorted(_aliases.names)]


def __getattr__(name: str) -> Any:
    matches = _aliases.select(name=name)
    if not matches:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if len(matches) > 1:
        raise ImportError(f"Multiple providers for {__name__}.{name}")
    return matches[name].load()
