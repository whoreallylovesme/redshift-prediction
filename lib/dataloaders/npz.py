from typing import Any

from lib.dataloaders.common import read_npz_pixels

_TEDDY_NAMES = (
    "PhotozDataset", "_center_crop_resize", "_compute_sample_weights", "_resolve_w1w2_path",
    "compute_stats", "get_npz_dataloaders", "load_npz", "parse_catalog",
)
__all__ = ["read_npz_pixels", *_TEDDY_NAMES]


def __getattr__(name: str) -> Any:
    if name in _TEDDY_NAMES:
        from photoz.data import npz as teddy

        return getattr(teddy, name)
    raise AttributeError(name)
