import asyncio
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch

import aiohttp

from tools.download_imgs import download_bricks, download_legacy_survey
from tools.download_imgs.workspace import brick_workspace, download_destination
from tools.euclid_q1.download_mosaics import download


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

    def test_partial_downloads_do_not_enter_shared_cache(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "band.fits"
            with self.assertRaises(TimeoutError), download_destination(path) as stream:
                stream.write(b"partial")
                raise TimeoutError
            self.assertEqual(list(path.parent.iterdir()), [])
            with download_destination(path) as first:
                first.write(b"slow response")
                with download_destination(path) as second:
                    second.write(b"published first")
                self.assertEqual(path.read_bytes(), b"published first")
            self.assertEqual(path.read_bytes(), b"published first")

    def test_mosaic_checks_length_before_publishing(self) -> None:
        class Reply(BytesIO):
            headers = {"Content-Length": "5"}

        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "image.fits"
            with patch("urllib.request.urlopen", return_value=Reply(b"abc")):
                with self.assertRaisesRegex(ValueError, "truncated"):
                    download("https://example.invalid/test", out, 1)
            self.assertEqual(list(out.parent.iterdir()), [])
            with patch("urllib.request.urlopen", return_value=Reply(b"abcde")):
                self.assertEqual(download("https://example.invalid/test", out, 1), 5)
            self.assertEqual(out.read_bytes(), b"abcde")

    def test_both_brick_downloaders_stream_and_discard_interrupted_responses(self) -> None:
        async def chunks(interrupted: bool) -> AsyncIterator[bytes]:
            yield b"first"
            if interrupted:
                raise aiohttp.ClientPayloadError("incomplete test response")
            yield b"second"

        with TemporaryDirectory() as tmp:
            for module in (download_bricks, download_legacy_survey):
                dest = Path(tmp) / module.__name__.split(".")[-1]
                session = MagicMock()
                response = session.get.return_value.__aenter__.return_value
                response.status = 200
                response.raise_for_status = MagicMock()
                with self.subTest(module=module.__name__), patch.object(module, "MAX_RETRIES", 1):
                    for interrupted in (True, False):
                        response.content.iter_chunked = MagicMock(return_value=chunks(interrupted))
                        ok = asyncio.run(
                            module._download_file(session, "fake", dest, asyncio.Semaphore(1))
                        )
                        self.assertEqual(ok, not interrupted)
                        self.assertEqual(dest.exists(), not interrupted)
                    self.assertEqual(dest.read_bytes(), b"firstsecond")
