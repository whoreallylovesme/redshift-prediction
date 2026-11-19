import ast
import subprocess
import sys
from pathlib import Path
from unittest import TestCase


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


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

    def test_layer_dependencies_point_in_one_direction(self) -> None:
        root = Path(__file__).parents[1]
        rules = {
            "lib": ("photoz",),
            "photoz/models": ("photoz.data", "photoz.training", "photoz.methods"),
            "photoz/data": ("photoz.models", "photoz.training", "photoz.methods"),
        }
        violations = []
        for directory, forbidden in rules.items():
            for path in (root / directory).rglob("*.py"):
                for module in imported_modules(path):
                    if any(module == name or module.startswith(name + ".") for name in forbidden):
                        violations.append(f"{path.relative_to(root)} -> {module}")
        self.assertEqual(violations, [])

    def test_generic_capsnet_modules_import_without_catalog_adapters(self) -> None:
        subprocess.run([sys.executable, "-c", """
import sys

sys.modules["photoz.data.hayat"] = None
sys.modules["photoz.data.teddy_capsnet"] = None
import photoz.methods.capsnet
import photoz.training.capsnet
import photoz.training.labels
"""], check=True)

    def test_catalog_specific_training_is_not_hidden_in_generic_modules(self) -> None:
        root = Path(__file__).parents[1]
        forbidden = ("photoz.data.hayat", "photoz.data.teddy_capsnet")
        for name in (
            "photoz/training/capsnet.py",
            "photoz/training/labels.py",
            "photoz/methods/capsnet.py",
        ):
            modules = imported_modules(root / name)
            self.assertFalse(any(module in forbidden for module in modules), name)
