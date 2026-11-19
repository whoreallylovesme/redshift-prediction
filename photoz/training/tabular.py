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
from torch.utils.data import DataLoader, TensorDataset

from lib.train.lightning import predict_batches, single_device_trainer
from lib.utils.utils import load_yaml_as, pick_device
from photoz.data.euclid_transforms import BANDS
from photoz.models.euclid_clip import DESCRIPTOR_DIM, TABLE_BANDS
from photoz.models.tabular import Architecture, TableRegressor
from photoz.training.clip_epochs import ShuffledRowBatches


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
    obs = np.stack((np.arcsinh(np.where(mask, flux, 0)), np.log1p(np.where(mask, error, 0))), -1)
    samples = obs[fit][mask[fit]]
    mean, std = samples.mean(0), samples.std(0)
    std[std <= 0] = 1
    values = np.where(mask[..., None], (obs - mean) / std, 0).astype(np.float32)
    return Projection(
        ids,
        z.astype(np.float64),
        fit,
        valid,
        torch.from_numpy(values),
        torch.from_numpy(mask),
        torch.as_tensor(descriptor, dtype=torch.float32),
        mean,
        std,
    )


class TableModule(L.LightningModule):
    def __init__(self, data: Projection, recipe: TableRecipe) -> None:
        super().__init__()
        self.model = TableRegressor(recipe.architecture, recipe.seed)
        self.data = data
        self.recipe = recipe
        self.target = torch.from_numpy(np.log1p(data.truth).astype(np.float32))
        self.register_buffer("descriptor", data.descriptor, persistent=False)
        self.best = math.inf
        self.best_state: dict[str, Tensor] | None = None
        self.selected_epoch = 0
        self.history: list[dict[str, Any]] = []
        self.valid_errors: list[np.ndarray] = []

    def configure_optimizers(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.model.parameters(), lr=self.recipe.lr, weight_decay=self.recipe.weight_decay
        )

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        return batch

    def on_train_epoch_start(self) -> None:
        epoch, warm = self.current_epoch + 1, self.recipe.warmup_epochs
        if epoch <= warm:
            factor = epoch / warm
        else:
            factor = 0.5 * (1 + math.cos(math.pi * (epoch - warm) / (self.recipe.epochs - warm)))
        self.lr = self.recipe.lr * factor
        self.train_sum, self.train_rows = 0.0, 0
        for group in self.trainer.optimizers[0].param_groups:
            group["lr"] = self.lr

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
        loss = (self.predict_rows(batch) - self.target[batch].to(self.device)).square().mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite table regression loss")
        self.train_sum += float(loss.detach()) * len(batch)
        self.train_rows += len(batch)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.valid_errors.clear()

    def validation_step(self, batch: tuple[Tensor], batch_idx: int) -> None:
        rows = batch[0]
        errors = self.predict_rows(rows).double().cpu() - self.target[rows].double()
        self.valid_errors.append(errors.square().numpy())

    def on_validation_epoch_end(self) -> None:
        valid = float(np.concatenate(self.valid_errors).mean())
        if not math.isfinite(valid):
            raise ValueError("Nonfinite validation log-MSE")
        self.log("val/log_mse", valid)
        train = self.train_sum / self.train_rows
        self.log("train/log_mse", train)
        epoch = self.current_epoch + 1
        self.history.append(dict(epoch=epoch, lr=self.lr, train_logMSE=train, FVALID_logMSE=valid))
        if valid < self.best:
            self.best, self.selected_epoch = valid, epoch
            self.best_state = {
                k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
            }


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
    single_device_trainer(device, recipe.epochs, gradient_clip=5.0).fit(module, train, valid)
    if module.best_state is None:
        raise ValueError("No validation checkpoint was selected")
    torch.save(module.model.state_dict(), out / "last_state.pt")
    torch.save(module.best_state, out / "selected_state.pt")
    module.model.load_state_dict(module.best_state, strict=True)
    module.to(device).eval()
    report = {
        "architecture": recipe.architecture,
        "seed": recipe.seed,
        "selected_epoch": module.selected_epoch,
        "validation_log_mse": module.best,
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
