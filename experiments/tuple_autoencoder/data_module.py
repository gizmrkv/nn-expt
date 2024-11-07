import itertools
from typing import Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class TupleDataModule(L.LightningDataModule):
    def __init__(
        self,
        tuple_size: int,
        range_size: int,
        batch_size: int,
        sample_size: int,
        *,
        num_workers: int = 4,
    ):
        super().__init__()
        self.tuple_size = tuple_size
        self.range_size = range_size
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.num_workers = num_workers

        self.dataset: TensorDataset | None = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            data = torch.randint(
                low=0,
                high=self.range_size,
                size=(self.sample_size, self.tuple_size),
            )
            self.dataset = TensorDataset(data, data)

    def train_dataloader(self) -> DataLoader[Tuple[torch.Tensor, ...]]:
        assert self.dataset is not None
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
