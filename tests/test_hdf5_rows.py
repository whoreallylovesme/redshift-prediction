import pickle
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import h5py
import numpy as np
from torch.utils.data import DataLoader


class HDF5RowsTest(TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "measurements.h5"
        self.ids = np.arange(2**60, 2**60 + 3, dtype=np.int64)
        self.measurements = np.arange(18, dtype=np.float64).reshape(3, 2, 3)

    def write_measurements(self) -> None:
        with h5py.File(self.path, "w") as file:
            file["survey/run/id"] = self.ids
            file["survey/run/measurements"] = self.measurements

    def test_nested_array_preserves_row_shape_and_dtype(self) -> None:
        from lib.dataloaders.hdf5_rows import HDF5Rows

        rows = HDF5Rows(self.path, "survey/run/measurements")
        self.addCleanup(rows.close)
        self.write_measurements()
        self.assertEqual(rows.shape, self.measurements.shape)
        self.assertEqual(len(rows), len(self.measurements))
        for i in (2, 0, 2):
            np.testing.assert_array_equal(rows[i], self.measurements[i], strict=True)

    def test_empty_array_has_zero_rows_but_group_and_scalar_have_no_row_axis(self) -> None:
        from lib.dataloaders.hdf5_rows import HDF5Rows

        with h5py.File(self.path, "w") as file:
            file["sensors/empty"] = np.empty((0, 3))
            file["sensors/scalar"] = 7
            file["sensors/null"] = h5py.Empty("f4")
        self.assertEqual(len(HDF5Rows(self.path, "sensors/empty")), 0)
        for key in ("sensors", "sensors/scalar", "sensors/null"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "row axis"):
                HDF5Rows(self.path, key)[0]

    def test_open_reader_can_be_pickled_and_reopened_after_close(self) -> None:
        from lib.dataloaders.hdf5_rows import HDF5Rows

        self.write_measurements()
        rows = HDF5Rows(self.path, "survey/run/id")
        self.addCleanup(rows.close)
        self.assertEqual(rows[2], self.ids[2])
        restored = pickle.loads(pickle.dumps(rows))
        self.addCleanup(restored.close)
        rows.close()
        for reader in (restored, rows):
            np.testing.assert_array_equal(reader[1], self.ids[1], strict=True)

    def test_spawn_worker_reads_ids_after_parent_access(self) -> None:
        from lib.dataloaders.hdf5_rows import HDF5Rows

        self.write_measurements()
        rows = HDF5Rows(self.path, "survey/run/id")
        self.addCleanup(rows.close)
        self.assertEqual(rows[0], self.ids[0])
        loader = DataLoader(
            rows,
            num_workers=1,
            batch_size=2,
            multiprocessing_context="spawn",
            collate_fn=np.stack,
            timeout=30,
        )
        ids = np.concatenate(list(loader))
        np.testing.assert_array_equal(ids, self.ids, strict=True)
        self.assertEqual(rows[2], self.ids[2])

    def test_generic_reader_works_without_photoz(self) -> None:
        self.write_measurements()
        source = (
            "import sys\n"
            "sys.modules['photoz'] = None\n"
            "from lib.dataloaders.hdf5_rows import HDF5Rows\n"
            "rows = HDF5Rows(sys.argv[1], 'survey/run/id')\n"
            "assert [int(rows[i]) for i in range(len(rows))] == [2**60 + i for i in range(3)]\n"
            "rows.close()\n"
        )
        run = subprocess.run(
            [sys.executable, "-c", source, str(self.path)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_hayat_tuple_preserves_subset_labels_and_cropped_pixels(self) -> None:
        from photoz.data.hayat import HayatDataset

        pixels = np.arange(3 * 5 * 8 * 7, dtype=np.int16).reshape(3, 5, 8, 7)
        with h5py.File(self.path, "w") as file:
            file["images"] = pixels
        indices = np.array([2, 0])
        redshifts = np.array([0.25, 0.5], dtype=np.float32)
        morphologies = np.array([np.nan, 1], dtype=np.float32)
        dataset = HayatDataset(
            self.path,
            indices,
            z_labels=redshifts,
            morph_labels=morphologies,
            crop_size=4,
            image_scale=2,
        )
        self.assertEqual(dataset.h5_path, self.path)
        for i, physical_row in enumerate(indices):
            image, redshift, morphology = dataset[i]
            expected = pixels[physical_row, :, 2:6, 1:5].astype(np.float32) / 2
            np.testing.assert_array_equal(image.numpy(), expected, strict=True)
            np.testing.assert_array_equal(redshift.numpy(), redshifts[i], strict=True)
            np.testing.assert_array_equal(morphology.numpy(), morphologies[i], strict=True)
