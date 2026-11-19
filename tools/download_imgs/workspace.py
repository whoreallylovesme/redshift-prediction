from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory


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
