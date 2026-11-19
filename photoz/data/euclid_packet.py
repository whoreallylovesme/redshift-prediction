from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from lib.models.euclid_clip import CLIPModel, ITNetwork
from lib.utils.utils import check

KEYS = (
    "image",
    "image_available",
    "table_values",
    "table_mask",
    "table_available",
    "spectrum",
    "spectrum_mask",
    "spectrum_metadata",
    "spectrum_available",
)
IT_ARGS = (
    "image",
    "table_values",
    "table_mask",
    "table_descriptor",
    "image_available",
    "table_available",
)
SSL_EPOCHS = 12
FINE_EPOCHS = 40
SSL_BATCH = 256
FINE_BATCH = 128
INFER_BATCH = 512
LR = 3e-4
WD = 1e-4
CLIP_NORM = 5.0
WARMUP_EPOCHS = 5
HOLDOUT_SEED = 20261027
HOLDOUT_FRAC = 0.10
TABLE_SLOTS = 14

type Rows = NDArray[np.integer] | Sequence[int]


def reset_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def state_copy(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def select_rows(dataset: h5py.Dataset, indices: Rows) -> NDArray[Any]:
    rows = np.asarray(indices, dtype=np.int64)
    if rows.ndim != 1 or len(rows) != len(np.unique(rows)):
        raise ValueError("one unique physical row per batch")
    if np.any(rows < 0) or np.any(rows >= len(dataset)):
        raise IndexError("batch rows outside dataset")
    order = np.argsort(rows, kind="stable")
    inverse = np.argsort(order, kind="stable")
    selected: NDArray[Any] = dataset[rows[order]][inverse]
    return selected


class Packet:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.f = h5py.File(self.path, "r", rdcc_nbytes=16_000_000, rdcc_nslots=1009)
        self.ids: NDArray[np.int64] = self.f["object_id"][:]
        self.fields: NDArray[np.integer] = self.f["field_code"][:]
        self.roles: NDArray[np.integer] = self.f["role_bitmask"][:]
        check(
            len(self.ids) == len(np.unique(self.ids)),
            "packet ids must be unique after physical-component handling",
        )
        self.indices: dict[str, NDArray[np.int64]] = {
            s: self.f[s + "_indices"][:] for s in ("train", "valid", "test", "ssl")
        }
        for s, bit in (("train", 2), ("valid", 4), ("test", 8), ("ssl", 1)):
            i = self.indices[s]
            inside = not np.any(i < 0) and not np.any(i >= len(self.ids))
            check(len(i) == len(np.unique(i)) and inside, "invalid split indices")
            check(np.all((self.roles[i] & bit) != 0), "split and role bits disagree")
        check(
            np.all(self.fields[self.indices["train"]] == 1)
            and np.all(self.fields[self.indices["valid"]] == 1),
            "supervised train/valid must be F only",
        )
        check(np.all(self.fields[self.indices["test"]] == 0), "north is exclusively test")
        check(np.isin(self.fields[self.indices["ssl"]], [1, 2]).all(), "SSL contains N rows")
        for a, b in (
            ("train", "valid"),
            ("train", "test"),
            ("valid", "test"),
            ("ssl", "valid"),
            ("ssl", "test"),
        ):
            check(not np.intersect1d(self.indices[a], self.indices[b]).size, "split overlap")
        check(
            {k for k in self.f.keys() if k.endswith("_z")} == {"train_z", "valid_z"},
            "only F training/validation labels belong in the model packet",
        )
        self.z: dict[str, NDArray[np.floating]] = {
            s: self.f[s + "_z"][:] for s in ("train", "valid")
        }
        for s, z in self.z.items():
            check(
                z.shape == (len(self.indices[s]),) and np.isfinite(z).all() and np.all(z >= 0),
                "invalid F labels",
            )
        for s in ("train", "valid"):
            rows = self.indices[s]
            usable = select_rows(self.f["table_available"], rows).astype(bool)
            observed = select_rows(self.f["table_mask"], rows).astype(bool).sum(axis=1)
            check(
                np.all(usable & (observed >= 2)),
                "every F fitting/validation row needs at least two measured table bands",
            )
        self.descriptor: NDArray[np.floating] = self.f["table_descriptor"][:]
        check(
            self.descriptor.shape == (TABLE_SLOTS, 18) and np.isfinite(self.descriptor).all(),
            "physical filter descriptor schema",
        )

    def close(self) -> None:
        self.f.close()


def holdout_tiles(
    packet: Packet,
) -> tuple[NDArray[np.int64], NDArray[np.int64], list[int], NDArray[np.integer]]:
    ssl = packet.indices["ssl"]
    owner: NDArray[np.integer] = packet.f["owner_tile"][:]
    rng = np.random.default_rng(HOLDOUT_SEED)
    held: list[int] = []
    for fc in (1, 2):
        tiles = np.unique(owner[ssl][packet.fields[ssl] == fc])
        n_held = max(1, round(HOLDOUT_FRAC * len(tiles)))
        held += rng.choice(tiles, n_held, replace=False).tolist()
    mask = np.isin(owner[ssl], held)
    return ssl[~mask], ssl[mask], sorted(int(t) for t in held), owner


def d4(image: Tensor, codes: Tensor) -> Tensor:
    if image.ndim != 4 or image.shape[-1] != image.shape[-2] or codes.shape != (len(image),):
        raise ValueError("square image batch and one D4 code per object")
    if torch.any((codes < 0) | (codes > 7)):
        raise ValueError("D4 code outside 0..7")
    transformed = torch.empty_like(image)
    for code in range(8):
        selected = codes == code
        if not torch.any(selected):
            continue
        x = torch.rot90(image[selected], code % 4, dims=(-2, -1))
        transformed[selected] = x.flip(-1) if code >= 4 else x
    return transformed


def schedule_lr(
    epoch: int, epochs: int, base_lr: float = LR, warmup: int = WARMUP_EPOCHS
) -> float:
    if not 1 <= epoch <= epochs or epochs <= warmup:
        raise ValueError("schedule epoch outside bounds")
    if epoch <= warmup:
        return base_lr * epoch / warmup
    return float(base_lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (epochs - warmup))))


def masked(b: dict[str, Tensor], keep: Tensor) -> dict[str, Tensor]:
    m = b["table_mask"] & keep
    b["table_mask"] = m
    b["table_available"] = b["table_available"] & (m.sum(1) >= 2)
    return b


def load_it_from_clip(path: Path) -> ITNetwork:
    state = torch.load(path, map_location="cpu")
    clip = CLIPModel(ITNetwork())
    clip.load_state_dict(state, strict=True)
    return clip.it_network
