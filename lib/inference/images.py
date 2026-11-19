from pathlib import Path
from typing import cast

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from lib.config import PytorchTrainConfig
from lib.config.schemas import ImageCatalogConfig
from lib.dataloaders.images import ImageInput, Images, catalog
from lib.train.lightning import single_device_trainer
from lib.train.regression import scalar_predictions
from lib.utils.estimator_loader import load_estimator
from lib.utils.utils import pick_device


class ImagePredictor(L.LightningModule):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def predict_step(
        self, batch: tuple[ImageInput, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        value = self.model(batch[0])
        pred = value[0] if isinstance(value, tuple) else value
        return scalar_predictions(pred, len(batch[1]))


def predict_catalog(
    cfg: PytorchTrainConfig,
    spec: ImageCatalogConfig,
    weights: Path,
    root: Path,
    labels: Path,
    out: Path,
    id_column: str | None,
    held_out: bool = False,
) -> None:
    rows = catalog(labels, spec)
    key = id_column or spec.id_column
    if key in {"y_true", "y_pred"}:
        raise ValueError("Image output ID must be distinct from y_true and y_pred")
    with np.load(weights.parent / "image_normalization.npz", allow_pickle=False) as norm:
        membership = None
        if "train_stems" in norm and "validation_stems" in norm:
            stems = set(rows[spec.id_column].astype(str))
            membership = {
                split: len(stems.intersection(norm[f"{split}_stems"]))
                for split in ("train", "validation")
            }
            if held_out and any(membership.values()):
                raise ValueError(f"Held-out image IDs overlap model development IDs: {membership}")
        elif held_out:
            raise ValueError("Cannot verify held-out images: weights lack split membership")
        stats = (norm["mean"], norm["std"])
        saved_features = norm["features"].tolist() if "features" in norm else []
        if saved_features != spec.features:
            raise ValueError("Inference feature columns differ from training feature order")
        table_stats = (norm["feature_mean"], norm["feature_std"]) if saved_features else None
    data = Images(root, rows, spec, stats, cfg.crop_size, cfg.out_size, table_stats)
    estimator = load_estimator(
        cfg.estimator, expected_type=torch.nn.Module, params=cfg.estimator_params,
        from_checkpoint=True,
    )
    model = cast(torch.nn.Module, estimator)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    frame = rows[[spec.id_column]].rename(columns={spec.id_column: key}).copy()
    if spec.target_column in rows:
        frame["y_true"] = rows[spec.target_column]
    prediction = np.empty(0, dtype=np.float32)
    if len(data):
        loader = DataLoader(data, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        trainer = single_device_trainer(pick_device(cfg.device), 1)
        values = trainer.predict(ImagePredictor(model), dataloaders=loader, return_predictions=True)
        if values is None:
            raise RuntimeError("Lightning did not return image predictions")
        prediction = torch.cat(cast(list[torch.Tensor], values)).cpu().numpy()
    frame["y_pred"] = prediction
    frame.attrs["prediction_id_column"] = key
    if membership is not None:
        frame.attrs["image_split_overlap"] = membership
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as stream:
        frame.to_parquet(stream, index=False)
