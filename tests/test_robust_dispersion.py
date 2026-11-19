from unittest import TestCase

import numpy as np

from lib.metrics.dispersion import scaled_mad
from photoz.data.euclid_transforms import SPECTRUM_LENGTH, UNIT_CGS, transform_spectrum
from photoz.data.euclid_translation import impute_ebv
from photoz.data.euclid_zero_points import gate
from photoz.data.euclid_zero_points import nmad as calibration_nmad
from photoz.metrics import nmad as redshift_nmad


class DispersionTests(TestCase):
    def test_missing_extinction_uses_valid_tile_median_then_global_median(self) -> None:
        tiles = np.array([2, 2, 2, 4, 4, 1, 1])
        values = np.array([0.04, -1, 0.08, np.inf, np.nan, 0.2, 0.4])
        filled, missing = impute_ebv(values, tiles)
        np.testing.assert_array_equal(filled, [0.04, 0.06, 0.08, 0.14, 0.14, 0.2, 0.4])
        np.testing.assert_array_equal(missing, [False, True, False, True, True, False, False])
        empty, mask = impute_ebv(np.array([np.inf, -1.0]), np.array([1, 2]))
        self.assertTrue(np.isnan(empty).all())
        self.assertTrue(mask.all())

    def test_calibration_requires_high_signal_in_all_eight_bands(self) -> None:
        signal = np.ones((4, 8), dtype=bool)
        signal[2, 7] = False
        signal[3, 0] = False
        prediction = np.ones((4, 8))
        prediction[0, :4] = [0, np.nan, np.inf, -1]
        expected = np.zeros_like(signal)
        expected[0, 4:] = True
        expected[1] = True
        np.testing.assert_array_equal(gate(prediction, signal), expected)

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
