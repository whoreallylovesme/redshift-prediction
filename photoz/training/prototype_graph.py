"""Inductive graph density regression from frozen, training-only prototypes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from lib.train.lightning import predict_batches
from lib.train.likelihood import LikelihoodModule, row_loader
from lib.utils.utils import load_yaml_as
from photoz.models.density import piecewise_uniform_nll, posterior_quantiles
from photoz.models.prototype_graph import ContextMode, PrototypeGraphDensity
from photoz.training.likelihood import LikelihoodRecipe, partition


class GraphRecipe(LikelihoodRecipe):
    context: ContextMode = "graph"


class GraphModule(LikelihoodModule):
    def __init__(self, graph: PrototypeGraphDensity, recipe: GraphRecipe) -> None:
        super().__init__(graph, recipe)
        self.edges = graph.head.z_edges

    def nll(self, batch: list[Tensor], *, training: bool) -> Tensor:
        embedding, index, weight, z = batch
        pred = self.model(embedding, index, weight)
        return piecewise_uniform_nll(pred.logits, z, self.edges)

    def predict_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        pred = self.model(*batch[:3])
        return torch.from_numpy(
            posterior_quantiles(pred.bin_mass.detach().cpu().numpy(), self.edges)
        )


def run_method(config_path: Path, out_dir: Path) -> Path:
    recipe = load_yaml_as(str(config_path), GraphRecipe)
    with np.load(recipe.projection, allow_pickle=False) as data:
        ids, train, valid = data["object_id"], data["fit_indices"], data["valid_indices"]
        z, embeddings, edges = data["z"], data["embedding"], data["z_edges"]
        bank_rows, nb, nb_weights = data["bank_rows"], data["bank_neighbors"], data["bank_weights"]
        ix, weights = data["query_neighbors"], data["query_weights"]
        permutation = data["context_permutation"] if recipe.context == "shuffled" else None
    partition(ids, train, valid)
    if (
        bank_rows.dtype != np.int64
        or bank_rows.ndim != 1
        or len(np.unique(bank_rows)) != len(bank_rows)
        or not np.isin(bank_rows, train).all()
    ):
        raise ValueError("bank_rows must contain distinct training-row indices")
    if (
        z.shape != (len(ids),)
        or z.dtype.kind not in "fiu"
        or not np.isfinite(z).all()
        or embeddings.shape != (len(ids), 64)
        or embeddings.dtype != np.float32
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Provide a finite reference z and 64 float32 embedding values per object")
    if (
        ix.shape != (len(ids), 4)
        or weights.shape != ix.shape
        or ix.dtype != np.int64
        or weights.dtype != embeddings.dtype
    ):
        raise ValueError("Each query requires four int64 bank indices and four float32 weights")
    torch.manual_seed(recipe.seed)
    graph = PrototypeGraphDensity(
        torch.from_numpy(embeddings[bank_rows]),
        torch.from_numpy(nb),
        torch.from_numpy(nb_weights),
        edges,
        context_mode=recipe.context,
        context_permutation=None if permutation is None else torch.from_numpy(permutation),
    )
    dataset = TensorDataset(
        torch.from_numpy(embeddings),
        torch.from_numpy(ix),
        torch.from_numpy(weights),
        torch.from_numpy(z.astype(np.float64)),
    )
    module = GraphModule(graph, recipe)
    val_loader = row_loader(dataset, valid, recipe)
    module.fit_selected(
        out_dir,
        row_loader(dataset, train, recipe, shuffle=True),
        val_loader,
        {
            "context": recipe.context,
            "prototype_rows": len(bank_rows),
            "fit_rows": len(train),
            "valid_rows": len(valid),
        },
    )
    quantiles = torch.cat(predict_batches(module, val_loader, module.device)).cpu().numpy()
    frame = pd.DataFrame(
        {
            "object_id": ids[valid],
            "y_true": z[valid],
            "y_pred": quantiles[:, 1],
            "q16": quantiles[:, 0],
            "q84": quantiles[:, 2],
        }
    )
    path = out_dir / "predictions.parquet"
    frame.to_parquet(path, index=False)
    return path
