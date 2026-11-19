import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from lib.dataloaders.batches import ShuffledRowBatches
from lib.train.lightning import fit_best, predict_batches
from lib.utils.utils import load_yaml_as, pick_device
from photoz.data.euclid_transforms import BANDS, transform_table
from photoz.data.euclid_transforms import fit_table as fit_table_normalizer
from photoz.models.euclid_clip import DESCRIPTOR_DIM, TABLE_BANDS
from photoz.models.tabular import Architecture, TableRegressor
from photoz.training.clip_epochs import OptimizerSchedule, scheduled_adamw


class TableRecipe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    projection: Path
    architecture: Architecture
    seed: int = 20261025
    epochs: int = Field(default=40, gt=0)
    batch_size: int = Field(default=128, gt=0)
    lr: float = Field(default=3e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    warmup_epochs: int = Field(default=5, ge=0)
    device: str = "cpu"


@dataclass
class Projection:
    ids: NDArray[np.int64]
    truth: NDArray[np.float64]
    fit: NDArray[np.int64]
    valid: NDArray[np.int64]
    values: Tensor
    mask: Tensor
    descriptor: Tensor
    mean: NDArray[np.float64]
    std: NDArray[np.float64]


def read_projection(path: Path) -> Projection:
    with np.load(path, allow_pickle=False) as src:
        ids, field, z = src["object_id"], src["field_code"], src["z"]
        roles, bands = src["role_bitmask"], src["bands"]
        fit, valid = src["fit_indices"], src["valid_indices"]
        mask, descriptor = src["table_mask"], src["table_descriptor"]
        flux = np.asarray(src["raw_flux_ujy"], dtype=np.float64)
        error = np.asarray(src["raw_error_ujy"], dtype=np.float64)
    n = len(ids)
    if ids.dtype != np.int64 or ids.shape != (n,) or len(np.unique(ids)) != n:
        raise ValueError("Projection requires one unique int64 object ID per row")
    if field.shape != (n,) or not np.all(field == 1):
        raise ValueError("The F-only experiment requires field_code 1 for every row")
    if z.shape != (n,) or not np.isfinite(z).all() or np.any(z < 0):
        raise ValueError("Expected one finite nonnegative reference redshift per row")
    if any(ix.dtype != np.int64 or ix.ndim != 1 or not len(ix) for ix in (fit, valid)):
        raise ValueError("Training and validation indices must be nonempty int64 vectors")
    if not np.array_equal(np.sort(np.concatenate((fit, valid))), np.arange(n)):
        raise ValueError("Training and validation must partition every row exactly once")
    if roles.shape != (n,) or roles.dtype.kind not in "iu":
        raise ValueError("One integer role mask is required for each projected row")
    if ((roles[fit] & 2) == 0).any() or (roles[fit] & 12).any():
        raise ValueError("Fit indices are inconsistent with native training membership")
    if ((roles[valid] & 4) == 0).any() or (roles[valid] & 11).any():
        raise ValueError("Validation indices are inconsistent with native validation membership")
    if tuple(bands.astype(str)) != BANDS:
        raise ValueError("Native photometry slots must follow the physical band order")
    if mask.dtype != np.bool_ or mask.shape != (n, TABLE_BANDS) or (mask.sum(1) < 2).any():
        raise ValueError("Expected at least two measured native bands per object")
    if descriptor.shape != (TABLE_BANDS, DESCRIPTOR_DIM) or not np.isfinite(descriptor).all():
        raise ValueError("Expected finite physical descriptors for all native bands")
    if flux.shape != mask.shape or error.shape != mask.shape:
        raise ValueError("Photometry and uncertainty arrays must match the native mask")
    if not (np.isfinite(flux[mask]) & np.isfinite(error[mask]) & (error[mask] > 0)).all():
        raise ValueError("Observed measurements require finite flux and positive finite error")
    normalizer = fit_table_normalizer(flux[fit], error[fit], mask[fit])
    values = transform_table(flux, error, mask, normalizer)
    return Projection(
        ids,
        z.astype(np.float64),
        fit,
        valid,
        torch.from_numpy(values),
        torch.from_numpy(mask),
        torch.as_tensor(descriptor, dtype=torch.float32),
        np.asarray(normalizer.mean),
        np.asarray(normalizer.std),
    )


class TableModule(L.LightningModule):
    def __init__(self, data: Projection, recipe: TableRecipe) -> None:
        super().__init__()
        self.model = TableRegressor(recipe.architecture, recipe.seed)
        self.data = data
        self.recipe = recipe
        self.target = torch.from_numpy(np.log1p(data.truth).astype(np.float32))
        self.register_buffer("descriptor", data.descriptor, persistent=False)
        self.history: list[dict[str, Any]] = []

    def configure_optimizers(self) -> OptimizerSchedule:
        return scheduled_adamw(
            self.model.parameters(), self.recipe.epochs, lr=self.recipe.lr,
            weight_decay=self.recipe.weight_decay, warmup=self.recipe.warmup_epochs,
        )

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        return batch

    def on_train_epoch_start(self) -> None:
        self.lr = float(self.trainer.optimizers[0].param_groups[0]["lr"])

    def predict_rows(self, rows: Tensor) -> Tensor:
        prediction: Tensor = self.model(
            self.data.values[rows].to(self.device),
            self.data.mask[rows].to(self.device),
            self.descriptor,
        )
        return prediction

    def predict_step(self, batch: tuple[Tensor], batch_idx: int) -> Tensor:
        return self.predict_rows(batch[0]).float()

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss = F.mse_loss(self.predict_rows(batch), self.target[batch].to(self.device))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite table regression loss")
        self.log("train/log_mse", loss.detach().double(), on_step=False, on_epoch=True,
                 batch_size=len(batch))
        return loss

    def validation_step(self, batch: tuple[Tensor], batch_idx: int) -> None:
        rows = batch[0]
        loss = F.mse_loss(
            self.predict_rows(rows).double(), self.target[rows].to(self.device).double()
        )
        self.log("val/log_mse", loss, batch_size=len(rows))

    def on_train_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics
        valid = float(metrics["val/log_mse"])
        if not math.isfinite(valid):
            raise ValueError("Nonfinite validation log-MSE")
        self.history.append(dict(
            epoch=self.current_epoch + 1, lr=self.lr,
            train_logMSE=float(metrics["train/log_mse"]), FVALID_logMSE=valid,
        ))

def fit_table(data: Projection, recipe: TableRecipe, out: Path) -> TableModule:
    out.mkdir(parents=True, exist_ok=False)
    module = TableModule(data, recipe)
    device = pick_device(recipe.device)
    train = ShuffledRowBatches(data.fit, recipe.seed, recipe.batch_size, drop_last=False).loader()
    valid = DataLoader(
        TensorDataset(torch.from_numpy(data.valid)),
        batch_size=recipe.batch_size,
        generator=torch.Generator(),
    )
    epoch, score = fit_best(
        module, train, valid, device, recipe.epochs, monitor="val/log_mse",
        gradient_clip=5.0, export_dir=out,
    )
    module.to(device).eval()
    report = {
        "architecture": recipe.architecture,
        "seed": recipe.seed,
        "selected_epoch": epoch,
        "validation_log_mse": score,
        "history": module.history,
        "normalizer_mean": data.mean.tolist(),
        "normalizer_std": data.std.tolist(),
        "train_rows": len(data.fit),
        "validation_rows": len(data.valid),
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return module


@torch.inference_mode()
def predict_table(module: TableModule, rows: NDArray[np.int64]) -> pd.DataFrame:
    loader = DataLoader(TensorDataset(torch.from_numpy(rows)), batch_size=module.recipe.batch_size,
                        generator=torch.Generator())
    results = predict_batches(module, loader, module.device)
    log_z = torch.cat(results).cpu().numpy() if results else np.empty(0, dtype=np.float32)
    return pd.DataFrame(
        {
            "object_id": module.data.ids[rows],
            "y_true": module.data.truth[rows],
            "pred_log1pz": log_z.astype(np.float64),
            "y_pred": np.expm1(log_z.astype(np.float64)),
        }
    )


def run_method(config_path: Path, out_dir: Path) -> Path:
    recipe = load_yaml_as(str(config_path), TableRecipe)
    data = read_projection(recipe.projection)
    module = fit_table(data, recipe, out_dir)
    result = predict_table(module, data.valid)
    path = out_dir / "predictions.parquet"
    result.to_parquet(path, index=False)
    return path
