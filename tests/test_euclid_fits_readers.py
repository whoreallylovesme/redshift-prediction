from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import numpy as np
from astropy.io import fits

from photoz.data.euclid_mosaics import open_mosaic
from tools.euclid_q1 import cut_stamps, hetdex
from tools.euclid_q1.select_pretrain_objects import ERROR, FLUX, read_catalog_rows, read_header


class EuclidFitsReaderTests(TestCase):
    def test_full_and_sparse_mosaics_preserve_stored_pixels(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "mosaic.fits"
            for dtype, peak in (
                (np.uint8, 255), (np.int16, 32767), (np.uint32, 2**32 - 1), (np.float32, None)
            ):
                pixels = np.arange(81, dtype=dtype).reshape(9, 9)
                if peak is not None:
                    pixels[4, 5] = peak
                else:
                    pixels.view(np.uint32)[4, 5] = 0x7FC0ABCD
                for primary in (True, False):
                    with self.subTest(dtype=dtype, primary=primary):
                        image = fits.PrimaryHDU(pixels) if primary else fits.ImageHDU(pixels)
                        if dtype is np.int16:
                            image.header["BLANK"] = int(pixels[4, 5])
                        hdus = [image] if primary else [fits.PrimaryHDU(), image]
                        fits.HDUList(hdus).writeto(path, overwrite=True)
                        full, header = hetdex.read_plane(path)
                        self.assertIs(hetdex.read_plane, cut_stamps.load_plane)
                        self.assertEqual(full.dtype, dtype)
                        self.assertEqual(full.tobytes(), pixels.tobytes())
                        with open_mosaic(path, sparse=True) as (section, original):
                            stamp = section[3:7, 4:8].astype(dtype, copy=False)
                            self.assertEqual(stamp.tobytes(), pixels[3:7, 4:8].tobytes())
                            self.assertEqual(original["BITPIX"], header["BITPIX"])

    def test_euclid_mosaics_reject_non_native_scaling(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "scaled.fits"
            image = fits.PrimaryHDU(np.zeros((4, 4), np.int16))
            image.header["BSCALE"] = 0.125
            image.writeto(path)
            with self.assertRaisesRegex(ValueError, "unscaled"):
                hetdex.read_plane(path)

    def test_binary_table_offsets_preserve_ids_and_raw_float_words(self) -> None:
        ids = np.array([2**60 + 1, 13, 2**60 + 3], np.int64)
        words = np.array([0x80000000, 0x7F80ABCD, 0x3F800000], np.uint32)
        columns = [
            fits.Column(name="FLAGS", format="9X", array=np.zeros((3, 9), dtype=bool)),
            fits.Column(name="OBJECT_ID", format="1K", array=ids),
        ]
        for i, name in enumerate(FLUX + ERROR):
            columns.append(fits.Column(name=name, format="E", array=np.roll(words, i).view("f4")))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.fits"
            table = fits.BinTableHDU.from_columns(columns)
            table.writeto(path)
            rows = np.array([2, 0, 1, 0])
            got, flux, error = read_catalog_rows(path, rows)
            np.testing.assert_array_equal(got, ids[rows])
            for values, offset in ((flux, 0), (error, len(FLUX))):
                for j in range(len(FLUX)):
                    expected = np.roll(words, j + offset)[rows]
                    np.testing.assert_array_equal(values.view("u4")[:, j], expected)
            raw = path.read_bytes()
            path.write_bytes(raw.replace(b"'1K      '", b"'1Kwrong '"))
            with self.assertRaisesRegex(ValueError, "unsupported TFORM"):
                read_catalog_rows(path, rows)
            table.header["TSCAL3"] = 2
            table.writeto(path, overwrite=True)
            with self.assertRaisesRegex(ValueError, "unscaled scalar"):
                read_catalog_rows(path, rows)

    def test_fits_header_rejects_non_ascii_and_truncation(self) -> None:
        hdr = fits.Header()
        hdr["NOTE"] = "abc"
        raw = hdr.tostring().encode("ascii")
        for invalid in (raw.replace(b"abc", b"\xffbc"), raw[:-1], raw.replace(b"END ", b"BAD ")):
            with self.assertRaises(ValueError):
                read_header(BytesIO(invalid))
