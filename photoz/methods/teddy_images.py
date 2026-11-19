from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import lightning as L
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, IterableDataset

from lib.config import PytorchTrainConfig
from lib.train.lightning import predict_batches
from lib.utils.estimator_loader import load_estimator
from lib.utils.log import get_logger
from lib.utils.utils import pick_device
from photoz.data.npz import compute_stats, load_npz, parse_catalog

logger = get_logger()


class ImageBatches(IterableDataset):
    def __init__(
        self,
        files: list[Path],
        cfg: PytorchTrainConfig,
        mean: NDArray[np.floating],
        std: NDArray[np.floating],
    ) -> None:
        self.files = files
        self.cfg = cfg
        self.mean, self.std = mean, std

    def __len__(self) -> int:
        return (len(self.files) + self.cfg.batch_size - 1) // self.cfg.batch_size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for start in range(0, len(self.files), self.cfg.batch_size):
            paths = self.files[start : start + self.cfg.batch_size]
            images, positions, reasons = [], [], [""] * len(paths)
            for index, path in enumerate(paths):
                try:
                    image = load_npz(
                        path,
                        self.mean,
                        self.std,
                        crop_size=self.cfg.crop_size,
                        out_size=self.cfg.out_size,
                    )
                except (ValueError, OSError, EOFError, KeyError) as exc:
                    reasons[index] = f"{type(exc).__name__}: {exc}"
                else:
                    images.append(image)
                    positions.append(index)
            yield {
                "ids": [path.stem for path in paths],
                "positions": positions,
                "reasons": reasons,
                "images": torch.cat(images) if images else None,
            }


class TeddyPredictor(L.LightningModule):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> list[dict[str, Any]]:
        values = np.full(len(batch["ids"]), np.nan)
        if batch["images"] is not None:
            output = self.model(batch["images"])
            pred = output[0] if isinstance(output, tuple) else output
            n = len(batch["positions"])
            if pred.shape not in {(n,), (n, 1)}:
                raise ValueError("Expected one prediction per Teddy image")
            values[batch["positions"]] = pred.reshape(-1).cpu().numpy()
        return [
            {"id": key, "y_pred": value, "status": reason or "ok"}
            for key, value, reason in zip(batch["ids"], values, batch["reasons"], strict=True)
        ]


def predict_teddy(
    model_path: str,
    cfg: PytorchTrainConfig,
    data_dir: str,
    catalog: str | None,
    out: str,
    id_column: str | None = None,
) -> None:
    estimator = load_estimator(
        cfg.estimator,
        expected_type=torch.nn.Module,
        params=cfg.estimator_params,
        from_checkpoint=True,
    )
    model = cast(torch.nn.Module, estimator)
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    paths = sorted(Path(data_dir).glob("*.npz"))
    if not paths:
        raise ValueError(f"No .npz files in {data_dir}")
    stats = Path(model_path).parent / "image_normalization.npz"
    if stats.is_file():
        with np.load(stats, allow_pickle=False) as saved:
            mean, std = saved["mean"], saved["std"]
    else:
        mean, std = compute_stats(sorted(Path(cfg.norm_from or cfg.data_dir).glob("*.npz")))
    targets = parse_catalog(catalog) if catalog is not None else {}
    batches = ImageBatches(paths, cfg, mean, std)
    results = predict_batches(
        TeddyPredictor(model), DataLoader(batches, batch_size=None), pick_device(cfg.device)
    )
    rows = [r for batch in results for r in batch]
    frame = pd.DataFrame(rows)
    if targets:
        frame["y_true"] = frame["id"].map(targets)
    if id_column:
        frame = frame.rename(columns={"id": id_column})
    else:
        frame = frame.loc[frame["status"] == "ok"].drop(columns=["id", "status"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
