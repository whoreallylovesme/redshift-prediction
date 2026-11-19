from collections.abc import Iterator

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class ShuffledRowBatches(IterableDataset[torch.Tensor]):
    def __init__(self, rows: NDArray[np.int64], seed: int, size: int, drop_last: bool) -> None:
        if size < 1:
            raise ValueError("Batch size must be positive")
        self.rows = rows
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.drop_last = drop_last

    def __len__(self) -> int:
        return (len(self.rows) + (0 if self.drop_last else self.size - 1)) // self.size

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        if worker is None:
            order = self.rng.permutation(self.rows)
            batches = range(len(self))
        else:
            loader_seed = worker.seed - worker.id
            epoch_seed = int(self.rng.integers(0, 2**63))
            order = np.random.default_rng([loader_seed, epoch_seed]).permutation(self.rows)
            batches = range(worker.id, len(self), worker.num_workers)
        for batch in batches:
            start = batch * self.size
            yield torch.from_numpy(order[start : start + self.size])

    def loader(self) -> DataLoader:
        if not len(self):
            raise ValueError("Training requires at least one batch")
        return DataLoader(self, batch_size=None, generator=torch.Generator())
