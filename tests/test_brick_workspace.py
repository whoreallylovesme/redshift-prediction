import asyncio
from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

import aiohttp
import pandas as pd
from click.testing import CliRunner

from tools.download_imgs import download_bricks, survey
from tools.download_imgs.workspace import brick_workspace, download_destination
from tools.euclid_q1.download_mosaics import download


class BrickWorkspaceTests(TestCase):
    def test_sdss_retry_policy_and_shared_cache(self) -> None:
        first = 2**60
        cached = pd.DataFrame({"objid": [first], "ra": [1.0], "dec": [2.0]})
        requested = [first + i for i in range(202)]
        with TemporaryDirectory() as tmp:
            cache = Path(tmp) / "coords.parquet"
            cached.to_parquet(cache)
            with (
                patch.object(survey, "_try_sdss", return_value=None) as query,
                patch.object(survey.time, "sleep") as sleep,
            ):
                self.assertEqual(survey.fetch_coordinates(requested[:2], cache).objid[0], first)
            self.assertEqual(query.call_count, 5)
            waits = [call(2**i) for i in range(4)]
            self.assertEqual(sleep.call_args_list, waits)

            new = pd.DataFrame({"objid": requested[1:], "ra": 3.0, "dec": 4.0})
            responses = [new.iloc[:200], new.iloc[200:]]
            with patch.object(survey, "_try_sdss", side_effect=responses) as query:
                result = survey.fetch_coordinates(requested, cache)
                self.assertEqual(result.objid.tolist(), requested)
                replay = survey.fetch_coordinates(requested, cache)
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

    def test_no_matching_brick_is_an_empty_mapping(self) -> None:
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
        self.assertEqual(survey.map_objects_to_bricks(coords, bricks), {})
        result = survey.map_objects_to_bricks(coords.assign(ra=39.0), bricks)
        self.assertEqual(result, {"0390p040": [(1, 39.0, 4.0)]})

    def test_no_coverage_exits_before_downloading(self) -> None:
        with TemporaryDirectory() as tmp:
            ids = Path(tmp) / "ids.json"
            ids.write_text('{"A": [1]}')
            coords = pd.DataFrame({"objid": [1], "ra": [30.0], "dec": [3.0]})
            with (
                patch.object(survey, "fetch_coordinates", return_value=coords),
                patch.object(survey, "load_brick_index", return_value=pd.DataFrame()),
                patch.object(survey, "map_objects_to_bricks", return_value={}),
            ):
                args = ["--teddy-json", str(ids), "--happy-json", str(ids), "--outdir", tmp]
                result = CliRunner().invoke(
                    download_bricks.main, [*args, "--dataset", "teddy", "--split", "A", "--dry-run"]
                )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertIn("No coordinates matched", result.output)

    def test_truncated_brick_index_is_not_reused(self) -> None:
        class Reply(BytesIO):
            headers = {"Content-Length": "12"}

        with TemporaryDirectory() as tmp, patch.object(survey, "urlopen", return_value=Reply(b"x")):
            with self.assertRaisesRegex(ValueError, "Truncated"):
                survey.load_brick_index(Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_brick_download_streams_and_discards_interrupted_responses(self) -> None:
        async def chunks(interrupted: bool) -> AsyncIterator[bytes]:
            yield b"first"
            if interrupted:
                raise aiohttp.ClientPayloadError("incomplete test response")
            yield b"second"

        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "band.fits"
            session = MagicMock()
            response = session.get.return_value.__aenter__.return_value
            response.status = 200
            response.raise_for_status = MagicMock()
            with patch.object(survey, "MAX_RETRIES", 1):
                for interrupted in (True, False):
                    response.content.iter_chunked = MagicMock(return_value=chunks(interrupted))
                    ok = asyncio.run(
                        survey._download_file(session, "fake", dest, asyncio.Semaphore(1))
                    )
                    self.assertEqual(ok, not interrupted)
                    self.assertEqual(dest.exists(), not interrupted)
                self.assertEqual(dest.read_bytes(), b"firstsecond")
