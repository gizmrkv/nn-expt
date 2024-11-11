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
        train_ratio: float = 0.8,
        *,
        num_workers: int = 4,
    ):
        super().__init__()
        self.tuple_size = tuple_size
        self.range_size = range_size
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers

        self.train_dataset: TensorDataset | None = None
        self.valid_dataset: TensorDataset | None = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            combinations = list(
                itertools.product(range(self.range_size), repeat=self.tuple_size)
            )
            all_data = torch.tensor(combinations, dtype=torch.long)
            indexes = torch.randperm(len(all_data))
            all_data = all_data[indexes]

            train_size = int(self.sample_size * self.train_ratio)
            train_data = all_data[:train_size]
            valid_data = all_data[train_size:]

            if self.sample_size > len(train_data):
                train_data = torch.cat(
                    [
                        train_data,
                        train_data[
                            torch.randint(
                                high=len(train_data),
                                size=(self.sample_size - len(train_data),),
                            )
                        ],
                    ]
                )

            self.train_dataset = TensorDataset(train_data, train_data)
            self.valid_dataset = TensorDataset(valid_data, valid_data)

    def train_dataloader(self) -> DataLoader[Tuple[torch.Tensor, ...]]:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Tuple[torch.Tensor, ...]]:
        assert self.valid_dataset is not None
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class SortTupleDataModule(L.LightningDataModule):
    def __init__(
        self,
        tuple_size: int,
        range_size: int,
        batch_size: int,
        sample_size: int,
        train_ratio: float = 0.8,
        *,
        num_workers: int = 4,
    ):
        super().__init__()
        self.tuple_size = tuple_size
        self.range_size = range_size
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers

        self.train_dataset: TensorDataset | None = None
        self.valid_dataset: TensorDataset | None = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            combinations = list(
                itertools.product(range(self.range_size), repeat=self.tuple_size)
            )
            all_data = torch.tensor(combinations, dtype=torch.long)

            indexes = torch.randperm(len(all_data))
            all_data = all_data[indexes]

            sorted_data = torch.sort(all_data, dim=1)[0]

            train_size = int(self.sample_size * self.train_ratio)
            train_data = all_data[:train_size]
            train_sorted = sorted_data[:train_size]
            valid_data = all_data[train_size:]
            valid_sorted = sorted_data[train_size:]

            if self.sample_size > len(train_data):
                indices = torch.randint(
                    high=len(train_data),
                    size=(self.sample_size - len(train_data),),
                )
                train_data = torch.cat([train_data, train_data[indices]])
                train_sorted = torch.cat([train_sorted, train_sorted[indices]])

            self.train_dataset = TensorDataset(train_data, train_sorted)
            self.valid_dataset = TensorDataset(valid_data, valid_sorted)

    def train_dataloader(self) -> DataLoader[Tuple[torch.Tensor, ...]]:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Tuple[torch.Tensor, ...]]:
        assert self.valid_dataset is not None
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
