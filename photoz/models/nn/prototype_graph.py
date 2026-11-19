"""Inductive message passing over a fixed bank of training embeddings."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import torch
from numpy.typing import ArrayLike
from torch import nn

from lib.models.nn.density import DensityPrediction
from lib.models.nn.graph import ResidualMessage as ResidualPrototypeMessage
from lib.models.nn.graph import weighted_neighbors as _mean
from photoz.models.nn.density import PhotoZDensityHead

ContextMode = Literal["graph", "self_only", "shuffled"]
EMBEDDING_DIM = 64
PROTOTYPE_NEIGHBORS = 16
QUERY_NEIGHBORS = 4
MAX_PROTOTYPES = 4096
MAX_QUERIES = 4096


def _features(value: torch.Tensor, rows: int | None, name: str) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or value.ndim != 2
        or value.shape[1] != EMBEDDING_DIM
        or (rows is not None and len(value) != rows)
        or value.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite floating [rows,64] embeddings")


def _assignments(
    index: torch.Tensor,
    weight: torch.Tensor,
    *,
    rows: int,
    count: int,
    bank: torch.Tensor,
    exclude_self: bool,
) -> None:
    if (
        not isinstance(index, torch.Tensor)
        or not isinstance(weight, torch.Tensor)
        or index.dtype != torch.int64
        or index.shape != (rows, count)
        or weight.shape != index.shape
        or index.device != bank.device
        or weight.device != bank.device
        or weight.dtype != bank.dtype
        or (index < 0).any()
        or (index >= len(bank)).any()
        or not torch.isfinite(weight).all()
        or (weight < 0).any()
    ):
        raise ValueError("Exact bounded int64 neighbours and finite nonnegative weights required")
    total = weight.detach().sum(dim=1)
    if not torch.isfinite(total).all() or (total <= 0).any():
        raise ValueError("Every assignment row needs a finite strictly positive weight SUM")
    ordered = index.sort(dim=1).values
    if (ordered[:, 1:] == ordered[:, :-1]).any():
        raise ValueError("Every neighbour row must contain distinct indices")
    if exclude_self and (index == torch.arange(rows, device=bank.device)[:, None]).any():
        raise ValueError("Prototype neighbour rows exclude self; self has a separate skip")


class PrototypeGraphDensity(nn.Module):
    """Encode queries with two synchronous message layers and a 400-bin density head.

    Graph mode aggregates the indexed bank rows. The self-only control uses each
    query as its own context. The shuffled control applies one fixed permutation
    to query assignments. The bank and its directed edges are frozen buffers.
    """

    _bank: torch.Tensor
    _neighbour_index: torch.Tensor
    _neighbour_weight: torch.Tensor
    _context_permutation: torch.Tensor

    def __init__(
        self,
        bank: torch.Tensor,
        neighbour_index: torch.Tensor,
        neighbour_weight: torch.Tensor,
        z_edges: ArrayLike,
        *,
        context_mode: ContextMode,
        context_permutation: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        _features(bank, None, "Frozen prototype bank")
        if not 17 <= len(bank) <= MAX_PROTOTYPES:
            raise ValueError("Frozen bank requires 17 to 4096 rows with 16 distinct neighbors each")
        _assignments(
            neighbour_index,
            neighbour_weight,
            rows=len(bank),
            count=PROTOTYPE_NEIGHBORS,
            bank=bank,
            exclude_self=True,
        )
        if context_mode not in ("graph", "self_only", "shuffled"):
            raise ValueError("Unknown fixed context mode")
        if context_mode == "shuffled":
            if (
                not isinstance(context_permutation, torch.Tensor)
                or context_permutation.dtype != torch.int64
                or context_permutation.device != bank.device
                or context_permutation.shape != (len(bank),)
                or not torch.equal(
                    context_permutation.sort().values, torch.arange(len(bank), device=bank.device)
                )
            ):
                raise ValueError(
                    "Shuffled control requires one explicit fixed bank-index bijection"
                )
            permutation = context_permutation
        else:
            if context_permutation is not None:
                raise ValueError("Only shuffled mode accepts a context permutation")
            permutation = torch.arange(len(bank), device=bank.device)
        self.context_mode = context_mode
        self.register_buffer("_bank", bank.detach().clone())
        self.register_buffer("_neighbour_index", neighbour_index.detach().clone())
        self.register_buffer("_neighbour_weight", neighbour_weight.detach().clone())
        self.register_buffer("_context_permutation", permutation.detach().clone())
        self.message_layers = nn.ModuleList(
            [ResidualPrototypeMessage(), ResidualPrototypeMessage()]
        )
        self.head = PhotoZDensityHead(EMBEDDING_DIM, z_edges, hidden_dim=64, dropout=0.0)
        if len(self.head.z_edges) != 401:
            raise ValueError(
                "This fixed prototype module requires exactly 400 physical-redshift bins"
            )
        self.to(device=bank.device, dtype=bank.dtype)
        self._expected_frozen_sha256 = self._actual_frozen_sha256()

    def _actual_frozen_sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self._bank,
            self._neighbour_index,
            self._neighbour_weight,
            self._context_permutation,
        ):
            digest.update(json.dumps(list(value.shape)).encode())
            dtype = torch.float64 if value.is_floating_point() else torch.int64
            canonical = value.detach().to(device="cpu", dtype=dtype).contiguous().numpy()
            digest.update(canonical.astype("<f8" if dtype == torch.float64 else "<i8").tobytes())
        return digest.hexdigest()

    def config_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "embedding_dim": EMBEDDING_DIM,
            "prototype_rows": len(self._bank),
            "prototype_neighbours": PROTOTYPE_NEIGHBORS,
            "query_neighbours": QUERY_NEIGHBORS,
            "max_queries_per_call": MAX_QUERIES,
            "message_layers": 2,
            "context_mode": self.context_mode,
            "weight_rule": "finite nonnegative / positive row SUM, never degree",
            "direction": "receiver row i <- explicitly listed source neighbours j",
            "synchronous_rule": "q1 <- bank0; q2 <- bank1; no query feedback",
            "upstream_query_embeddings_and_assignments": "detached frozen external inputs",
            "frozen_inputs_sha256": self._expected_frozen_sha256,
            "parameter_dtype": str(next(self.parameters()).dtype),
            "head": self.head.config_dict(),
        }

    def get_extra_state(self) -> dict[str, Any]:
        if self._actual_frozen_sha256() != self._expected_frozen_sha256:
            raise ValueError("Frozen prototype buffers changed since construction")
        return self.config_dict()

    def set_extra_state(self, state: Any) -> None:
        if (
            state != self.config_dict()
            or self._actual_frozen_sha256() != self._expected_frozen_sha256
        ):
            raise ValueError("Prototype checkpoint mode, configuration or frozen bank mismatch")

    def prototype_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Defensive bank0 copy and differentiable bank1; neither uses a query."""
        context = _mean(self._bank, self._neighbour_index, self._neighbour_weight)
        state1 = self.message_layers[0](self._bank, context)
        return self._bank.clone(), state1

    def encode_queries(
        self,
        query: torch.Tensor,
        prototype_index: torch.Tensor,
        prototype_weight: torch.Tensor,
    ) -> torch.Tensor:
        _features(query, None, "Frozen query embeddings")
        first = next(self.parameters())
        if len(query) > MAX_QUERIES or query.device != first.device or query.dtype != first.dtype:
            raise ValueError("Query chunk must fit 4096 rows and match module dtype/device")
        _assignments(
            prototype_index,
            prototype_weight,
            rows=len(query),
            count=QUERY_NEIGHBORS,
            bank=self._bank,
            exclude_self=False,
        )
        own, weight = query.detach(), prototype_weight.detach()
        if self.context_mode == "self_only":
            for layer in self.message_layers:
                own = layer(own, own)
            return own
        states = self.prototype_states()
        index = (
            self._context_permutation[prototype_index]
            if self.context_mode == "shuffled"
            else prototype_index
        )
        for layer, bank_state in zip(self.message_layers, states, strict=True):
            own = layer(own, _mean(bank_state, index, weight))
        return own

    def forward(
        self,
        query: torch.Tensor,
        prototype_index: torch.Tensor,
        prototype_weight: torch.Tensor,
    ) -> DensityPrediction:
        result: DensityPrediction = self.head(
            self.encode_queries(query, prototype_index, prototype_weight)
        )
        return result
