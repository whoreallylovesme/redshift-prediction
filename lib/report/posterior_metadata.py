from itertools import pairwise
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

METADATA_KEY = b"regression.posterior.v1"


class PosteriorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    posterior_column: str = Field(min_length=1)
    posterior_grid: list[FiniteFloat] = Field(min_length=2)

    @field_validator("posterior_grid")
    @classmethod
    def increasing(cls, axis: list[float]) -> list[float]:
        if any(lo >= hi for lo, hi in pairwise(axis)):
            raise ValueError("Posterior grid must be strictly increasing")
        return axis

    @classmethod
    def read(cls, path: Path) -> "PosteriorMetadata":
        schema = pq.read_schema(path)
        raw = (schema.metadata or {}).get(METADATA_KEY)
        if raw is None:
            raise ValueError(f"Predictor did not supply posterior metadata in {path.name}")
        spec = cls.model_validate_json(raw)
        if spec.posterior_column not in schema.names:
            raise ValueError(f"Posterior column {spec.posterior_column!r} is missing")
        vector = schema.field(spec.posterior_column).type
        if not (
            pa.types.is_list(vector)
            or pa.types.is_large_list(vector)
            or pa.types.is_fixed_size_list(vector)
        ) or not pa.types.is_floating(vector.value_type):
            raise ValueError("Posterior column must contain lists of floating-point values")
        return spec
