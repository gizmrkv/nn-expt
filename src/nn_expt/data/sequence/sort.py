import itertools
from typing import Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class SequenceSortDataModule(L.LightningDataModule):
    def __init__(
        self,
        max_length: int,
        vocab_size: int,
        batch_size: int,
        *,
        train_ratio: float = 0.8,
        num_workers: int = 4,
        n_repeats: int = 1,
    ):
        super().__init__()
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.n_repeats = n_repeats

        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            combinations = list(
                itertools.product(range(self.vocab_size), repeat=self.max_length)
            )
            all_data = torch.tensor(combinations, dtype=torch.long)

            indexes = torch.randperm(len(all_data))
            all_data = all_data[indexes]

            sorted_data = torch.sort(all_data, dim=1)[0]

            train_size = int(len(all_data) * self.train_ratio)
            train_data = all_data[:train_size]
            train_sorted = sorted_data[:train_size]
            val_data = all_data[train_size:]
            val_sorted = sorted_data[train_size:]

            train_data = torch.repeat_interleave(train_data, self.n_repeats, dim=0)

            self.train_dataset = TensorDataset(train_data, train_sorted)
            self.val_dataset = TensorDataset(val_data, val_sorted)

    def train_dataloader(self) -> DataLoader[Tuple[torch.Tensor, ...]]:
        assert self.train_dataset is not None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader[Tuple[torch.Tensor, ...]]:
        assert self.val_dataset is not None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
