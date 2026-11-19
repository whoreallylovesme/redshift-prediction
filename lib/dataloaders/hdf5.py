from typing import Any

from lib.dataloaders.hdf5_rows import HDF5Rows

_HAYAT_NAMES = ("HDF5Dataset", "get_hdf5_dataloaders", "_worker_init_fn", "logger")
__all__ = ["HDF5Rows", *_HAYAT_NAMES]


def __getattr__(name: str) -> Any:
    if name in _HAYAT_NAMES:
        from photoz.data import hdf5 as legacy_hayat

        return getattr(legacy_hayat, name)
    raise AttributeError(name)
