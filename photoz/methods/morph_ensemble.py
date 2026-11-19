from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal, cast

import lightning as L
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, TensorDataset

from lib.train.lightning import single_device_trainer
from lib.utils.utils import load_yaml_as
from photoz.data.weak import exact_ids
from photoz.models.capsnet import CapsNet, logistic_to_z

SEEDS = (201, 202, 203, 204)
REF_IDS = (
    2672438317641790906,
    2672486961639939402,
    2672431058642451487,
    2684888071681001110,
    2684838616681055197,
    2684890888681271472,
)
REF_VALUES = {
    201: (
        0.15603968501091003,
        1.193023681640625,
        0.4018592834472656,
        0.16438846290111542,
        0.9800784587860107,
        0.27233076095581055,
    ),
    202: (
        0.11639996618032455,
        1.2653073072433472,
        0.32094651460647583,
        0.23837819695472717,
        1.177452802658081,
        0.4899325668811798,
    ),
    203: (
        0.12210424244403839,
        0.9939057230949402,
        0.5269342660903931,
        0.26548612117767334,
        1.1541801691055298,
        0.46130746603012085,
    ),
    204: (
        0.1793346107006073,
        0.9014732241630554,
        0.3037007749080658,
        0.3789783716201782,
        1.1416517496109009,
        0.4770106077194214,
    ),
}
WEIGHT_HASHES = {
    201: "649fc75f7ce8a16bc2c1be868a66c9511aa3d86d350fc5e90cbd5ea2f245e620",
    202: "901ca1494cec2a165ce9000c83e5719b068bb43d3bd995530deb50018995a160",
    203: "ce21b10c437490584a25104b1e3590f40fd630b92f389cc964c052699dfa1f42",
    204: "2ba9d0b27052bae4dc19e8e90751452ffc3ff89d89d4b24716c63eec7ff1c997",
}
STAMP_HASHES = {
    102157954: "048fd6d2b56f4287de731614ceb1405386beab7b67249993d812c8fe20929bfd",
    102160336: "93b90001c4ec836cefcb90a58ab7ae41df793f8312f585399317c10c9fc2c219",
}
NORM_HASH = "2e292e8d6872ba8649eb05a7669e04be964696f1dd22149cdf2b88e6eaa97d6d"


def normalize_images(images: NDArray[Any], scales: NDArray[Any]) -> NDArray[np.float32]:
    if scales.shape != (4,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Four finite positive saved normalization scales are required")
    if images.ndim != 4 or images.shape[1:] != (4, 64, 64):
        raise ValueError("Expected images [N,4,64,64] in VIS,Y,J,H order")
    normalized = (
        np.arcsinh(images.astype(np.float32) / scales.astype(np.float32)[None, :, None, None]) / 3.0
    )
    if not np.isfinite(normalized).all():
        raise ValueError("Nonfinite image pixels; refusing silent replacement")
    return np.asarray(normalized, dtype=np.float32)


class PinnedFile(BaseModel):
    path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StampSet(BaseModel):
    manifest: PinnedFile
    split: str | None = None
    shards: list[PinnedFile] = Field(min_length=1)
    ids_key: str = "ids"
    images_key: str = "X"


class MorphRecipe(BaseModel):
    checkpoint_root: Path
    reference_dir: Path
    sets: dict[str, StampSet]
    device: Literal["cpu", "mps"] = "cpu"
    batch_size: int = Field(default=16, ge=1, le=64)


def verify(stream: Any, expected: str) -> None:
    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
        raise ValueError(f"Frozen artifact digest mismatch: {stream.name}")
    stream.seek(0)


def read_ids(root: Path, dataset: StampSet) -> pd.Index:
    with (root / dataset.manifest.path).open("rb") as stream:
        verify(stream, dataset.manifest.sha256)
        columns = ["object_id", "split"] if dataset.split is not None else ["object_id"]
        frame = pd.read_parquet(stream, columns=columns)
    if dataset.split is not None:
        frame = frame.loc[frame.split.eq(dataset.split)]
    exact_ids(frame, ("object_id",))
    ids = pd.Index(frame.object_id.to_numpy(np.int64))
    if ids.empty or not ids.is_unique:
        raise ValueError("A nonempty unique-ID frozen evaluation cohort is required")
    return ids


def load_network(root: Path, seed: int) -> tuple[CapsNet, NDArray[np.float32]]:
    directory = root / f"morph_{seed}"
    with (directory / "norm_scales.npy").open("rb") as norm:
        verify(norm, NORM_HASH)
        scales = np.asarray(np.load(norm, allow_pickle=False), dtype=np.float32)
    normalize_images(np.zeros((1, 4, 64, 64), np.float32), scales)
    model = CapsNet(in_channels=4, input_size=64, num_class=2, dim_capsule=16, routings=3)
    with (directory / "stage_b_best.pt").open("rb") as weights:
        verify(weights, WEIGHT_HASHES[seed])
        model.load_state_dict(torch.load(weights, weights_only=True, map_location="cpu"))
    return model, scales


class MorphPredictor(L.LightningModule):
    def __init__(self, network: CapsNet) -> None:
        super().__init__()
        self.network = network

    def predict_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        logits, _, _ = self.network(batch[0])
        return logistic_to_z(logits.cpu(), 0, 3)


def predict_shards(
    root: Path,
    shards: list[PinnedFile],
    ids_key: str,
    images_key: str,
    ids: pd.Index,
    model: MorphPredictor,
    scales: NDArray[np.float32],
    recipe: MorphRecipe,
) -> NDArray[np.float64]:
    result = np.full(len(ids), np.nan)
    trainer = single_device_trainer(torch.device(recipe.device), 1)
    for shard in shards:
        with (root / shard.path).open("rb") as stream:
            verify(stream, shard.sha256)
            with np.load(stream, allow_pickle=False) as data:
                source_ids = pd.Index(data[ids_key])
                exact_ids(pd.DataFrame({"object_id": source_ids}), ("object_id",))
                if not source_ids.is_unique:
                    raise ValueError("Duplicate object in prepared image shard")
                positions = source_ids.get_indexer(ids)
                matched = np.flatnonzero(positions >= 0)
                if not len(matched):
                    continue
                if np.isfinite(result[matched]).any():
                    raise ValueError("One evaluation object occurs in several prepared shards")
                images = data[images_key]
                if len(images) != len(source_ids):
                    raise ValueError("Stamp count differs from identity count")
                x = normalize_images(images[positions[matched]], scales)
        batches = trainer.predict(
            model,
            dataloaders=DataLoader(
                TensorDataset(torch.from_numpy(x)), batch_size=recipe.batch_size
            ),
            return_predictions=True,
        )
        if batches is None:
            raise ValueError("No morph predictions")
        values = np.concatenate([v.numpy() for v in cast(list[torch.Tensor], batches)])
        if values.shape != (len(matched),) or not np.isfinite(values).all():
            raise ValueError("Morph predictor requires one finite value per object and seed")
        result[matched] = values
    if not np.isfinite(result).all():
        raise ValueError("Frozen cohort has missing morph images")
    return result


def predict(config_path: Path, set_name: str, out_dir: Path) -> Path:
    recipe = load_yaml_as(str(config_path), MorphRecipe)
    if set_name not in recipe.sets:
        raise ValueError(f"Unknown morph evaluation set: {set_name}")
    root = config_path.parent
    spec = recipe.sets[set_name]
    ids = read_ids(root, spec)
    refs = [PinnedFile(path=Path(f"{tile}.npz"), sha256=sha) for tile, sha in STAMP_HASHES.items()]
    out_dir.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame({"object_id": ids})
    for seed in SEEDS:
        network, scales = load_network(root / recipe.checkpoint_root, seed)
        model = MorphPredictor(network)
        actual = predict_shards(
            root / recipe.reference_dir, refs, "ids", "X", pd.Index(REF_IDS), model, scales, recipe
        )
        if not np.allclose(actual, REF_VALUES[seed], rtol=1e-4, atol=2e-5):
            raise ValueError(f"Morph {seed}: original CPU reference check failed")
    for seed in SEEDS:
        network, scales = load_network(root / recipe.checkpoint_root, seed)
        model = MorphPredictor(network)
        frame[f"z_pred_{seed}"] = predict_shards(
            root, spec.shards, spec.ids_key, spec.images_key, ids, model, scales, recipe
        )
    values = frame[[f"z_pred_{seed}" for seed in SEEDS]]
    frame["y_pred"] = values.mean(axis=1, skipna=False)
    out = out_dir / "predictions.parquet"
    with out.open("xb") as stream:
        frame.to_parquet(stream, index=False)
    return out
