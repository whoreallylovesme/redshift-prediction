from unittest import TestCase

import numpy as np

from lib.metrics.dispersion import scaled_mad
from photoz.data.euclid_transforms import SPECTRUM_LENGTH, UNIT_CGS, transform_spectrum
from photoz.data.euclid_zero_points import nmad as calibration_nmad
from photoz.metrics import nmad as redshift_nmad


class DispersionTests(TestCase):
    def test_historical_factor_empty_and_nonfinite_values(self) -> None:
        values = np.array([-2.0, 1.0, 4.0, 8.0, 9.0])
        expected = 1.4826 * np.median(np.abs(values - np.median(values)))
        self.assertEqual(scaled_mad(values), expected)
        self.assertEqual(calibration_nmad(values), expected)
        self.assertEqual(redshift_nmad(np.zeros_like(values), values), expected)
        self.assertEqual(scaled_mad(np.array([5.0])), 0.0)
        self.assertTrue(np.isnan(scaled_mad(np.array([1.0, np.nan, 4.0]))))
        self.assertTrue(np.isnan(redshift_nmad(np.array([]), np.array([]))))
        self.assertIsNone(calibration_nmad(np.array([])))

    def test_spectrum_keeps_scale_and_centre_of_observed_bins(self) -> None:
        mask = np.arange(SPECTRUM_LENGTH) < 5
        flux = np.full(SPECTRUM_LENGTH, np.nan)
        values = np.array([-2.0, 1.0, 4.0, 8.0, 9.0]) * UNIT_CGS
        flux[mask] = values
        product = {
            "spectrum_valid": mask,
            "spectrum_present": np.array(True),
            "spectrum_flux_cgs": flux,
            "spectrum_error_cgs": np.full(SPECTRUM_LENGTH, UNIT_CGS),
            "spectrum_wavelength_angstrom": np.full(SPECTRUM_LENGTH, 15000.0),
        }
        encoded, valid, meta, present = transform_spectrum(product, UNIT_CGS)
        centre = np.median(values)
        scale = 1.4826 * np.median(np.abs(values - centre))
        self.assertTrue(present)
        np.testing.assert_array_equal(valid, mask)
        np.testing.assert_array_equal(encoded[mask, 0], ((values - centre) / scale).astype("f4"))
        np.testing.assert_array_equal(
            meta, np.array([np.arcsinh(centre / UNIT_CGS), np.log1p(scale / UNIT_CGS),
                            mask.mean()], dtype="f4"),
        )
