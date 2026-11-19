from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from lib.config import PytorchTrainConfig
from lib.config.schemas import ImageCatalogConfig
from lib.dataloaders.common import (
    center_crop,
    inverse_bin_weights,
    read_npz_pixels,
    resize_square,
    standardize_channels,
    subsample_indices,
)
from lib.dataloaders.dataset import exact_index

type ImageInput = torch.Tensor | dict[str, torch.Tensor]

type Stats = tuple[NDArray[np.float32], NDArray[np.float32]]
type FeatureStats = tuple[NDArray[np.float64], NDArray[np.float64]]


def catalog(path: Path, spec: ImageCatalogConfig) -> pd.DataFrame:
    if spec.id_column == spec.target_column:
        raise ValueError("Image identifier and target columns must differ")
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={spec.id_column: "string"})
    else:
        frame = pd.read_parquet(path)
    original = exact_index(frame[spec.id_column], spec.id_column)
    if spec.id_dtype == "int64" and len(original) and pd.api.types.is_integer_dtype(original.dtype):
        bounds = np.iinfo(np.int64)
        if int(original.min()) < bounds.min or int(original.max()) > bounds.max:
            raise ValueError("Image IDs exceed the signed int64 range")
    if spec.id_dtype:
        frame[spec.id_column] = frame[spec.id_column].astype(spec.id_dtype)
    ids = frame[spec.id_column]
    stems = ids.astype(str)
    if (
        ids.isna().any()
        or stems.duplicated().any()
        or any(Path(stem).name != stem or stem in {"", ".", ".."} for stem in stems)
    ):
        raise ValueError("Image IDs must be unique, nonempty file stems")
    return frame


def pixels(path: Path, key: str) -> NDArray[np.float32]:
    array = read_npz_pixels(path, key)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or not all(array.shape) or not np.isfinite(array).all():
        raise ValueError(f"{path}: expected a finite H×W or H×W×C image")
    return array


def channel_stats(paths: list[Path], key: str) -> Stats:
    if not paths:
        raise ValueError("Image training split is empty")
    scaler = StandardScaler()
    for path in paths:
        image = pixels(path, key)
        scaler.partial_fit(image.reshape(-1, image.shape[-1]))
    return scaler.mean_.astype(np.float32), np.sqrt(scaler.var_).astype(np.float32)


class Images(Dataset[tuple[ImageInput, torch.Tensor]]):
    def __init__(
        self,
        images: Path,
        rows: pd.DataFrame,
        spec: ImageCatalogConfig,
        stats: Stats,
        crop: int | None,
        size: int,
        table_stats: FeatureStats | None = None,
    ) -> None:
        self.features: NDArray[np.float32] | None = None
        if spec.features:
            if table_stats is None:
                raise ValueError("Multimodal input requires training feature statistics")
            values = rows[spec.features].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Tabular input features must be finite")
            mean, std = table_stats
            self.features = ((values - mean) / std).astype(np.float32)
        self.paths = [images / f"{item}.npz" for item in rows[spec.id_column]]
        self.targets = (
            rows[spec.target_column].to_numpy(dtype=np.float32)
            if spec.target_column in rows
            else np.full(len(rows), np.nan, dtype=np.float32)
        )
        self.key = spec.array_key
        self.mean, self.std = stats
        self.crop = crop
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[ImageInput, torch.Tensor]:
        path = self.paths[index]
        image = pixels(path, self.key)
        if image.shape[-1] != len(self.mean):
            raise ValueError(f"{path}: expected {len(self.mean)} input channels")
        if self.crop:
            h, w = image.shape[:2]
            if self.crop > min(h, w):
                raise ValueError(f"{path}: requested crop exceeds image dimensions")
            image = center_crop(image, self.crop)
        tensor = standardize_channels(image, self.mean, self.std)
        tensor = resize_square(tensor, self.size)
        target = torch.tensor(self.targets[index])
        if self.features is not None:
            return {"image": tensor, "features": torch.from_numpy(self.features[index])}, target
        return tensor, target


def image_loaders(
    cfg: PytorchTrainConfig, spec: ImageCatalogConfig, out: Path
) -> tuple[DataLoader, DataLoader]:
    if cfg.norm_from or cfg.physics_augment:
        raise ValueError("Generic image recipes use normalization fitted on training images")
    root = Path(cfg.data_dir)
    frame = catalog(Path(cfg.catalog), spec)
    if not np.isfinite(frame[spec.target_column].to_numpy(dtype=np.float64)).all():
        raise ValueError("Image training targets must be finite")
    if cfg.max_samples is not None and cfg.max_samples < len(frame):
        frame = frame.iloc[subsample_indices(len(frame), cfg.max_samples, cfg.seed)]
    train, valid = train_test_split(frame, train_size=cfg.train_ratio, random_state=cfg.seed)
    stats = channel_stats([root / f"{item}.npz" for item in train[spec.id_column]], spec.array_key)
    table_stats: FeatureStats | None = None
    norm: dict[str, np.ndarray] = {
        "mean": stats[0],
        "std": stats[1],
        "train_stems": np.asarray(train[spec.id_column].astype(str), dtype=np.str_),
        "validation_stems": np.asarray(valid[spec.id_column].astype(str), dtype=np.str_),
    }
    if spec.features:
        columns = train[spec.features].to_numpy(dtype=np.float64)
        if not np.isfinite(columns).all():
            raise ValueError("Training features must be finite")
        feature_mean = columns.mean(0)
        feature_std = columns.std(0)
        feature_std[feature_std == 0] = 1
        table_stats = feature_mean, feature_std
        norm.update(
            features=np.array(spec.features), feature_mean=feature_mean, feature_std=feature_std,
        )
    training = Images(root, train, spec, stats, cfg.crop_size, cfg.out_size, table_stats)
    validation = Images(root, valid, spec, stats, cfg.crop_size, cfg.out_size, table_stats)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "image_normalization.npz").open("xb") as stream:
        np.savez(stream, allow_pickle=False, **norm)
    sampler = None
    if cfg.weighted_sampling:
        weights = inverse_bin_weights(training.targets)
        sampler = WeightedRandomSampler(weights.tolist(), len(weights))
    return (
        DataLoader(
            training,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=sampler is None,
            sampler=sampler,
        ),
        DataLoader(validation, batch_size=cfg.batch_size, num_workers=cfg.num_workers),
    )
