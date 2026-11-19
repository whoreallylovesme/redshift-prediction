import asyncio
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import aiohttp
import numpy as np
import pandas as pd
from astropy.io import fits
from click.testing import CliRunner

from tools.download_imgs import download_bricks, download_cutouts, download_legacy_survey, survey
from tools.download_imgs.workspace import brick_workspace, download_destination
from tools.euclid_q1.download_mosaics import download


class BrickWorkspaceTests(TestCase):
    def test_interrupted_cutout_can_be_retried_and_keeps_the_original_padding(self) -> None:
        pixels = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
        with BytesIO() as stream:
            fits.PrimaryHDU(pixels).writeto(stream)
            response = MagicMock()
            response.get.return_value.__aenter__.return_value.status = 200
            response.get.return_value.__aenter__.return_value.raise_for_status = MagicMock()
            response.get.return_value.__aenter__.return_value.read.return_value = stream.getvalue()

        def interrupt(stream: BinaryIO, *, image: np.ndarray) -> None:
            stream.write(b"partial NPZ")
            raise OSError("disk full")

        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "1.npz"
            with patch.object(np, "savez_compressed", side_effect=interrupt):
                with self.assertRaisesRegex(OSError, "disk full"):
                    asyncio.run(download_cutouts.download_one_cutout(
                        response, asyncio.Semaphore(1), 1, 0.0, 0.0, dest,
                    ))
            self.assertEqual(list(dest.parent.iterdir()), [])
            self.assertTrue(asyncio.run(download_cutouts.download_one_cutout(
                response, asyncio.Semaphore(1), 1, 0.0, 0.0, dest,
            )))
            expected = np.zeros((224, 224, 4), dtype=np.float32)
            expected[:2, :3] = pixels.transpose(1, 2, 0)
            with np.load(dest) as result:
                np.testing.assert_array_equal(result["image"], expected)

    def test_sdss_retry_policies_and_shared_cache(self) -> None:
        first = 2**60
        cached = pd.DataFrame({"objid": [first], "ra": [1.0], "dec": [2.0]})
        requested = [first + i for i in range(202)]
        for module, attempts, backoff in [(download_bricks, 5, 1), (download_cutouts, 8, 2)]:
            with self.subTest(module=module.__name__), TemporaryDirectory() as tmp:
                cache = Path(tmp) / "coords.parquet"
                cached.to_parquet(cache)
                with (
                    patch.object(survey, "_try_sdss", return_value=None) as query,
                    patch.object(survey.time, "sleep") as sleep,
                ):
                    self.assertEqual(module.fetch_coordinates(requested[:2], cache).objid[0], first)
                self.assertEqual(query.call_count, attempts)
                waits = [call(backoff * 2**i) for i in range(attempts - 1)]
                self.assertEqual(sleep.call_args_list, waits)

                new = pd.DataFrame({"objid": requested[1:], "ra": 3.0, "dec": 4.0})
                responses = [new.iloc[:200], new.iloc[200:]]
                with patch.object(survey, "_try_sdss", side_effect=responses) as query:
                    result = module.fetch_coordinates(requested, cache)
                    self.assertEqual(result.objid.tolist(), requested)
                    replay = module.fetch_coordinates(requested, cache)
                    self.assertEqual(replay.objid.tolist(), requested)
                ids_sent = [
                    q.args[0].split(" IN (")[1][:-1].split(",") for q in query.call_args_list
                ]
                self.assertEqual(ids_sent, [list(map(str, requested[1:201])), [str(requested[-1])]])

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

    def test_no_matching_brick_is_an_empty_mapping_for_both_commands(self) -> None:
        coords = pd.DataFrame({"objid": [1], "ra": [358.2], "dec": [4.0]})
        bricks = pd.DataFrame(
            {
                "brickname": ["0390p040"],
                "ra1": [38.9],
                "ra2": [39.1],
                "dec1": [3.9],
                "dec2": [4.1],
            }
        )
        for module in (download_bricks, download_legacy_survey):
            self.assertEqual(module.map_objects_to_bricks(coords, bricks), {})
            result = module.map_objects_to_bricks(coords.assign(ra=39.0), bricks)
            self.assertEqual(result, {"0390p040": [(1, 39.0, 4.0)]})

    def test_no_coverage_exits_before_downloading_in_both_clis(self) -> None:
        with TemporaryDirectory() as tmp:
            ids = Path(tmp) / "ids.json"
            ids.write_text('{"A": [1]}')
            coords = pd.DataFrame({"objid": [1], "ra": [30.0], "dec": [3.0]})
            for module, flags in [
                (download_bricks, ["--dataset", "teddy", "--split", "A", "--dry-run"]),
                (download_legacy_survey, ["--datasets", "teddy", "--splits", "A"]),
            ]:
                with (
                    patch.object(module, "fetch_coordinates", return_value=coords),
                    patch.object(module, "load_brick_index", return_value=pd.DataFrame()),
                    patch.object(module, "map_objects_to_bricks", return_value={}),
                ):
                    args = ["--teddy-json", str(ids), "--happy-json", str(ids), "--outdir", tmp]
                    result = CliRunner().invoke(module.main, [*args, *flags])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("No coordinates matched", result.output)

    def test_truncated_brick_index_is_not_reused(self) -> None:
        class Reply(BytesIO):
            headers = {"Content-Length": "12"}

        with TemporaryDirectory() as tmp, patch.object(survey, "urlopen", return_value=Reply(b"x")):
            with self.assertRaisesRegex(ValueError, "Truncated"):
                survey.load_brick_index(Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

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
                with self.subTest(module=module.__name__), patch.object(survey, "MAX_RETRIES", 1):
                    for interrupted in (True, False):
                        response.content.iter_chunked = MagicMock(return_value=chunks(interrupted))
                        ok = asyncio.run(
                            module._download_file(session, "fake", dest, asyncio.Semaphore(1))
                        )
                        self.assertEqual(ok, not interrupted)
                        self.assertEqual(dest.exists(), not interrupted)
                    self.assertEqual(dest.read_bytes(), b"firstsecond")
