import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from pydantic import BaseModel, Field


class PinnedFile(BaseModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def open_verified(path: Path, expected_sha256: str) -> Iterator[BinaryIO]:
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            raise ValueError(f"File digest mismatch: {path}")
        stream.seek(0)
        yield stream
