from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.io.fits.hdu.image import Section
from numpy.typing import NDArray

type Plane = NDArray[Any] | Section


@contextmanager
def open_mosaic(path: str | Path, *, sparse: bool = False) -> Iterator[tuple[Plane, fits.Header]]:
    with ExitStack() as stack:
        stream = stack.enter_context(Path(path).open("rb"))
        hdus = stack.enter_context(fits.open(stream, memmap=False))
        index = 0 if hdus[0].header.get("NAXIS") == 2 else 1
        hdu = hdus[index]
        header = hdu.header.copy()
        bits = header["BITPIX"]
        zero = header.get("BZERO", 0)
        if header["NAXIS"] != 2 or header.get("BSCALE", 1) != 1 or (
            zero != 0 and not (bits > 8 and zero == 2 ** (bits - 1))
        ):
            raise ValueError(f"Expected an unscaled or unsigned 2D mosaic: {path}")
        # In integer maps BLANK is a stored flag word. Read it without FITS null substitution.
        if bits > 0 and zero == 0 and "BLANK" in header:
            stream.seek(0)
            raw = stack.enter_context(fits.open(stream, do_not_scale_image_data=True, memmap=False))
            hdu = raw[index]
        yield (hdu.section if sparse else hdu.data), header


def load_plane(path: str | Path) -> tuple[NDArray[Any], fits.Header]:
    with open_mosaic(path) as (plane, header):
        return np.asarray(plane, dtype=plane.dtype.newbyteorder("=")), header
