import os
from pathlib import Path
from typing import Any

import h5py
from torch.utils.data import Dataset


class HDF5Rows(Dataset[Any]):
    def __init__(self, path: str | Path, key: str, *, swmr: bool = False) -> None:
        self.path = Path(path)
        self.key = key
        self.swmr = swmr
        self._pid: int | None = None
        self._file: h5py.File | None = None
        self._dataset: h5py.Dataset | None = None

    def _dataset_in(self, source: h5py.File) -> h5py.Dataset:
        dataset = source[self.key]
        if not isinstance(dataset, h5py.Dataset) or not dataset.shape:
            raise ValueError(f"{self.key!r} must be an HDF5 dataset with a row axis")
        return dataset

    def _open(self) -> h5py.Dataset:
        pid = os.getpid()
        if self._dataset is None or self._pid != pid:
            self.close()
            source = h5py.File(self.path, "r", swmr=self.swmr)
            try:
                dataset = self._dataset_in(source)
            except BaseException:
                source.close()
                raise
            self._file = source
            self._dataset = dataset
            self._pid = pid
        return self._dataset

    @property
    def shape(self) -> tuple[int, ...]:
        if self._dataset is not None and self._pid == os.getpid():
            return tuple(self._dataset.shape)
        self.close()
        with h5py.File(self.path, "r", swmr=self.swmr) as source:
            return tuple(self._dataset_in(source).shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index: int) -> Any:
        return self._open()[index]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
        self._file = self._dataset = self._pid = None

    def __getstate__(self) -> dict[str, Any]:
        return {**vars(self), "_file": None, "_dataset": None, "_pid": None}
