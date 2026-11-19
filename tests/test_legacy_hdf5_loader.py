import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import ZipFile

import h5py
import numpy as np


class LegacyHDF5LoaderTest(TestCase):
    root = Path(__file__).resolve().parents[1]

    def check_loader(self, directory: Path, module_root: Path) -> None:
        with h5py.File(directory / "measurements.h5", "w") as file:
            file["images"] = np.arange(16, dtype=np.int16).reshape(4, 1, 2, 2)
        fixture = self.root / "tests/fixtures/legacy/hayat_val_90208ed.pkl"
        probe = """
import pickle
import sys
from pathlib import Path

import numpy as np

module_root = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(Path.cwd()), str(module_root)]
assert "lib" not in sys.modules and "photoz" not in sys.modules
with open(sys.argv[2], "rb") as stream:
    loader = pickle.load(stream)

from lib.dataloaders import hdf5
from lib.dataloaders.hdf5 import *
from photoz.data.hayat import get_hayat_dataloaders

assert not hasattr(hdf5, "Other")
assert HDF5Dataset is type(loader.dataset)
assert _worker_init_fn is loader.worker_init_fn
assert get_hdf5_dataloaders is get_hayat_dataloaders
for name in (hdf5.__name__, HDF5Rows.__module__, HDF5Dataset.__module__):
    assert Path(sys.modules[name].__file__).resolve().is_relative_to(module_root), name

assert loader.num_workers == 1
loader.multiprocessing_context = "spawn"
loader.timeout = 30
[(images, targets, morphologies)] = list(loader)
expected = np.arange(16, dtype=np.float32).reshape(4, 1, 2, 2)[[1, 3]] / 2
np.testing.assert_array_equal(images.numpy(), expected, strict=True)
np.testing.assert_array_equal(targets.numpy(), np.array([1, 3], dtype=np.float32), strict=True)
np.testing.assert_array_equal(morphologies.numpy(), np.ones(2, dtype=np.float32), strict=True)
"""
        subprocess.run(
            [sys.executable, "-I", "-c", probe, str(module_root), str(fixture)],
            cwd=directory,
            check=True,
            timeout=90,
        )

    def test_checkout_with_stale_metadata_on_search_path(self) -> None:
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            metadata = directory / "astro-0.1.0.dist-info"
            metadata.mkdir()
            (metadata / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: astro\nVersion: 0.1.0\n"
            )
            (metadata / "entry_points.txt").write_text(
                "[astro.legacy.lib.dataloaders.hdf5]\nOther = decimal:Decimal\n"
            )
            self.check_loader(directory, self.root)

    def test_built_wheel(self) -> None:
        with TemporaryDirectory() as temp:
            directory = Path(temp)
            project = directory / "project"
            project.mkdir()
            shutil.copyfile(self.root / "pyproject.toml", project / "pyproject.toml")
            for name in ("lib", "photoz", "runtime", "tools"):
                shutil.copytree(
                    self.root / name, project / name, ignore=shutil.ignore_patterns("__pycache__")
                )
            build = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    "from setuptools.build_meta import build_wheel; build_wheel('dist')",
                ],
                cwd=project,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            [wheel] = (project / "dist").glob("*.whl")
            installation = directory / "site-packages"
            with ZipFile(wheel) as archive:
                archive.extractall(installation)
            self.check_loader(directory, installation)
