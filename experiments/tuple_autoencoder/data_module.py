import itertools
from typing import Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class TupleDataModule(L.LightningDataModule):
    def __init__(self, tuple_size: int, range_size: int, batch_size: int):
        super().__init__()
        self.tuple_size = tuple_size
        self.range_size = range_size
        self.batch_size = batch_size

        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            all_combinations = torch.tensor(
                list(itertools.product(range(self.range_size), repeat=self.tuple_size))
            )
            indices = torch.randperm(len(all_combinations))

            train_size = int(len(all_combinations) * 0.8)
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            train_tensor = all_combinations[train_indices]
            val_tensor = all_combinations[val_indices]

            self.train_data = TensorDataset(train_tensor)
            self.val_data = TensorDataset(val_tensor)

    def train_dataloader(self) -> DataLoader[Tuple[torch.Tensor, ...]]:
        assert self.train_data is not None
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> DataLoader[Tuple[torch.Tensor, ...]]:
        assert self.val_data is not None
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self) -> DataLoader[Tuple[torch.Tensor, ...]]:
        assert self.test_data is not None
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4)
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=4)
