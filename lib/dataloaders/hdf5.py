import tomllib
from importlib.metadata import Distribution, EntryPoint, EntryPoints
from pathlib import Path
from typing import Any

from lib.dataloaders.hdf5_rows import HDF5Rows as HDF5Rows


def _legacy_aliases() -> EntryPoints:
    group = f"astro.legacy.{__name__}"
    root = Path(__file__).resolve().parents[2]
    for dist in Distribution.discover(name="astro", path=[str(root)]):
        if dist.read_text("WHEEL") is not None:
            return dist.entry_points.select(group=group)
    manifest = root / "pyproject.toml"
    if manifest.is_file():
        with manifest.open("rb") as stream:
            declarations = tomllib.load(stream).get("project", {}).get("entry-points", {})
        if group in declarations:
            return EntryPoints(
                EntryPoint(name, target, group) for name, target in declarations[group].items()
            )
    return EntryPoints(())


_aliases = _legacy_aliases()
__all__ = ["HDF5Rows", *sorted(_aliases.names)]


def __getattr__(name: str) -> Any:
    try:
        alias = _aliases[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    return alias.load()
