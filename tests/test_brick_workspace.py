from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from tools.download_imgs.workspace import brick_workspace


class BrickWorkspaceTests(TestCase):
    def test_cleanup_on_error_preserves_existing_downloads_and_unrelated_files(self) -> None:
        with TemporaryDirectory() as temp:
            parent = Path(temp)
            with brick_workspace(parent, keep=True) as cache:
                owned_by_user = cache / "existing.fits"
                owned_by_user.write_bytes(b"cached")
            with self.assertRaises(RuntimeError), brick_workspace(parent) as scratch:
                self.assertNotEqual(cache, scratch)
                (scratch / "partial.fits").write_bytes(b"interrupted")
                raise RuntimeError("network failure")
            self.assertFalse(scratch.exists())
            self.assertEqual(owned_by_user.read_bytes(), b"cached")
