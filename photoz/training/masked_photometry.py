"""Target-free masked-flux pretraining and observed-flux reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pydantic import Field
from torch import Tensor
from torch.utils.data import TensorDataset

from lib.train.lightning import predict_batches
from lib.train.likelihood import LikelihoodModule, row_loader
from lib.utils.utils import load_yaml_as
from photoz.models.masked_photometry import MaskedFluxEncoder, masked_gaussian_nll, sample_mask
from photoz.training.likelihood import LikelihoodRecipe, partition


class MaskedRecipe(LikelihoodRecipe):
    hidden_dim: int = Field(default=64, gt=0)
    heads: int = Field(default=4, gt=0)
    layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    mask_fraction: float = Field(default=0.35, gt=0, lt=1)


class MaskedModule(LikelihoodModule):
    model: MaskedFluxEncoder

    def __init__(self, recipe: MaskedRecipe, bands: tuple[str, ...]) -> None:
        model = MaskedFluxEncoder(
            bands,
            hidden_dim=recipe.hidden_dim,
            num_heads=recipe.heads,
            num_layers=recipe.layers,
            dropout=recipe.dropout,
        )
        super().__init__(model, recipe)
        self.bands = bands
        self.fraction = recipe.mask_fraction

    def on_fit_start(self) -> None:
        self.mask_rng = torch.Generator(device=self.device).manual_seed(self.recipe.seed + 1)

    def nll(self, batch: list[Tensor], *, training: bool) -> Tensor:
        flux, error, valid, hidden = batch
        if training:
            hidden = sample_mask(valid, fraction=self.fraction, generator=self.mask_rng)
        pred = self.model(flux, error, valid, masked=hidden, bands=self.bands)
        return masked_gaussian_nll(pred, flux, error, valid, hidden)

    def predict_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        flux, error, valid, hidden = batch
        pred = self.model(flux, error, valid, masked=hidden, bands=self.bands)
        return torch.stack(
            (pred.mean, (pred.intrinsic_variance + error.double().square()).sqrt()), dim=-1
        )


def run_method(config_path: Path, out_dir: Path) -> Path:
    recipe = load_yaml_as(str(config_path), MaskedRecipe)
    with np.load(recipe.projection, allow_pickle=False) as data:
        ids, train, valid = data["object_id"], data["fit_indices"], data["valid_indices"]
        bands = tuple(data["bands"].tolist())
        flux, error, observed = data["raw_flux_ujy"], data["raw_error_ujy"], data["table_mask"]
    partition(ids, train, valid)
    if any(array.shape != (len(ids), len(bands)) for array in (flux, error, observed)):
        raise ValueError("Flux, error and table_mask must have shape [objects, bands]")
    if observed.dtype != np.bool_ or any(arr.dtype.kind not in "fiu" for arr in (flux, error)):
        raise ValueError("Expected numeric photometry and a boolean table_mask")
    flux, error = flux.astype(np.float64), error.astype(np.float64)
    if not (
        np.isfinite(flux[observed]) & np.isfinite(error[observed]) & (error[observed] > 0)
    ).all():
        raise ValueError("Measured flux and error must be finite, with strictly positive error")
    scales = []
    for col, band in enumerate(bands):
        rows = train[observed[train, col]]
        if not len(rows):
            raise ValueError(f"No observed training flux for band {band}")
        scales.append(
            max(
                float(np.quantile(np.abs(flux[rows, col]), 0.9)), float(np.median(error[rows, col]))
            )
        )
    scale = np.array(scales)
    f = torch.from_numpy(np.where(observed, flux / scale, 0).astype(np.float32))
    e = torch.from_numpy(np.where(observed, error / scale, 0).astype(np.float32))
    mask = torch.from_numpy(observed)
    if not torch.isfinite(f).all() or not torch.isfinite(e).all() or (e[mask] <= 0).any():
        raise ValueError("Scaled photometry must fit float32 with positive measurement errors")
    eligible = observed.sum(1) >= 2
    train_rows, valid_rows = train[eligible[train]], valid[eligible[valid]]
    if not len(train_rows) or not len(valid_rows):
        raise ValueError("Each split needs rows with at least two observed bands")
    fixed = torch.zeros_like(mask)
    fixed[valid_rows] = sample_mask(
        mask[valid_rows],
        fraction=recipe.mask_fraction,
        generator=torch.Generator().manual_seed(recipe.seed + 3),
    )
    dataset = TensorDataset(f, e, mask, fixed)
    valid_loader = row_loader(dataset, valid_rows, recipe)
    torch.manual_seed(recipe.seed)
    module = MaskedModule(recipe, bands)
    module.fit_selected(
        out_dir,
        row_loader(dataset, train_rows, recipe, shuffle=True),
        valid_loader,
        {
            "bands": bands,
            "scale_ujy": scales,
            "encoder": module.model.config_dict(),
            "fit_rows": len(train_rows),
            "valid_rows": len(valid_rows),
            "ineligible_fit": len(train) - len(train_rows),
            "ineligible_valid": len(valid) - len(valid_rows),
        },
    )
    result = torch.cat(predict_batches(module, valid_loader, module.device)).cpu().numpy()
    row, column = np.nonzero(fixed[valid_rows].numpy())
    obj = valid_rows[row]
    pred = result[row, column, 0] * scale[column]
    sigma = result[row, column, 1] * scale[column]
    frame = pd.DataFrame(
        {
            "measurement_id": [f"{i}:{bands[c]}" for i, c in zip(ids[obj], column, strict=True)],
            "object_id": ids[obj],
            "band": np.asarray(bands)[column],
            "y_true": flux[obj, column],
            "y_pred": pred,
            "observed_sigma": sigma,
        }
    )
    path = out_dir / "predictions.parquet"
    frame.to_parquet(path, index=False)
    return path
