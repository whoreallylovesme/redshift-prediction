import subprocess
import sys
from unittest import TestCase


class LibraryImportTests(TestCase):
    def test_all_lib_modules_import_without_photoz(self) -> None:
        subprocess.run([sys.executable, "-c", """
import importlib
import pkgutil
import sys

sys.modules["photoz"] = None
import lib

for module in pkgutil.walk_packages(lib.__path__, "lib."):
    importlib.import_module(module.name)
"""], check=True)
