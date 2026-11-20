import hashlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_SHA256 = r"^[0-9a-f]{64}$"


class PinnedFile(BaseModel):
    path: Path
    sha256: str = Field(pattern=_SHA256)


class ArtifactFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str = Field(pattern=_SHA256)
    bytes: int = Field(ge=0)

    @field_validator("path")
    @classmethod
    def relative(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Artifact files require relative safe paths")
        return value


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    method: str
    kind: Literal["tabular", "lightning", "external"]
    estimator: str
    task: str
    code_sha256: str = Field(pattern=_SHA256)
    dataset_sha256: str | None = Field(default=None, pattern=_SHA256)
    features: list[str] = Field(default_factory=list)
    id_column: str | None = None
    target: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    files: list[ArtifactFile] = Field(min_length=1)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for source in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(source.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(source)))
    return digest.hexdigest()


def input_fingerprints(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): sha256_path(path) for path in paths}


def write_artifact_manifest(
    directory: Path,
    paths: Sequence[Path],
    *,
    method: str,
    kind: Literal["tabular", "lightning", "external"],
    estimator: str,
    task: str,
    code_sha256: str,
    dataset_sha256: str | None,
    features: list[str],
    id_column: str | None,
    target: str | None,
    config: dict[str, Any],
) -> ArtifactManifest:
    root = directory.resolve()
    entries = []
    for path in sorted(set(paths), key=lambda item: str(item)):
        source = path.resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Artifact file must belong to {root}: {source}")
        entries.append(
            ArtifactFile(
                path=source.relative_to(root).as_posix(),
                sha256=sha256_file(source),
                bytes=source.stat().st_size,
            )
        )
    manifest = ArtifactManifest(
        method=method,
        kind=kind,
        estimator=estimator,
        task=task,
        code_sha256=code_sha256,
        dataset_sha256=dataset_sha256,
        features=features,
        id_column=id_column,
        target=target,
        config=config,
        files=entries,
    )
    destination = root / "artifact.json"
    with NamedTemporaryFile("w", encoding="utf-8", dir=root, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(manifest.model_dump_json(indent=2) + "\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination)
    return manifest


def verify_artifact_manifest(directory: Path) -> ArtifactManifest:
    root = directory.resolve()
    manifest = ArtifactManifest.model_validate_json((root / "artifact.json").read_text())
    for entry in manifest.files:
        source = (root / entry.path).resolve()
        if not source.is_relative_to(root) or not source.is_file():
            raise ValueError(f"Missing artifact file: {entry.path}")
        if source.stat().st_size != entry.bytes or sha256_file(source) != entry.sha256:
            raise ValueError(f"Artifact file changed: {entry.path}")
    return manifest


@contextmanager
def open_verified(path: Path, expected_sha256: str) -> Iterator[BinaryIO]:
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
            raise ValueError(f"File digest mismatch: {path}")
        stream.seek(0)
        yield stream
