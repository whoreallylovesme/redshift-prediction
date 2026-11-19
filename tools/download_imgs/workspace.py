import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import IO


@contextmanager
def download_destination(path: Path) -> Iterator[IO[bytes]]:
    """Publish a complete local download without replacing a concurrently completed file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent) as temp:
        yield temp.file
        temp.flush()
        os.fsync(temp.fileno())
        try:
            os.link(temp.name, path)
        except FileExistsError:
            if not path.is_file():
                raise


@contextmanager
def brick_workspace(parent: Path, *, keep: bool = False) -> Iterator[Path]:
    parent.mkdir(parents=True, exist_ok=True)
    if keep:
        path = parent / "bricks"
        path.mkdir(exist_ok=True)
        yield path
    else:
        with TemporaryDirectory(dir=parent, prefix="brick-") as scratch:
            yield Path(scratch)
