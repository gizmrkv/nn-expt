import itertools
from typing import Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader, TensorDataset


class Seq2SeqDataModule(L.LightningDataModule):
    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        batch_size: int,
        *,
        train_ratio: float = 0.8,
        num_workers: int = 4,
        num_repeats: int = 1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_workers = num_workers
        self.num_repeats = num_repeats

        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            combinations = list(
                itertools.product(range(self.vocab_size), repeat=self.seq_len)
            )
            all_data = torch.tensor(combinations, dtype=torch.long)
            indexes = torch.randperm(len(all_data))
            all_data = all_data[indexes]

            train_size = int(len(all_data) * self.train_ratio)
            train_data = all_data[:train_size]
            val_data = all_data[train_size:]

            train_data = torch.repeat_interleave(train_data, self.num_repeats, dim=0)

            self.train_dataset = TensorDataset(train_data, train_data)
            self.val_dataset = TensorDataset(val_data, val_data)

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
