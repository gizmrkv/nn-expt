import datetime
import itertools
from pathlib import Path
from typing import Tuple

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from callback import PlotEmbeddingsCallback, PlotLinearCallback
from data_module import TupleDataModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from moviepy.editor import ImageSequenceClip
from torch.utils.data import DataLoader, TensorDataset

import wandb


class TupleAutoencoder(L.LightningModule):
    def __init__(
        self,
        tuple_length: int,
        range_size: int,
        embedding_dim: int,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tuple_length = tuple_length
        self.range_size = range_size
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay

        self.embedding = nn.Embedding(range_size, embedding_dim)
        self.linear = nn.Linear(tuple_length * embedding_dim, tuple_length * range_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)

        embedded = self.embedding(x)
        embedded_flat = embedded.view(batch_size, -1)
        output = self.linear(embedded_flat)
        output = output.view(batch_size, self.tuple_length, self.range_size)
        output = F.softmax(output, dim=-1)

        return output

    def _compute_loss(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch[0]
        output = self(x)

        loss = F.nll_loss(
            output.log().view(-1, self.range_size), x.view(-1), reduction="mean"
        )
        return loss / x.size(0)

    def _compute_metrics(self, x: torch.Tensor, output: torch.Tensor):
        entropy = -(output * torch.log(output + 1e-10)).sum(dim=-1)
        mean_entropy_per_position = entropy.mean(dim=0)
        mean_entropy = entropy.mean()

        sampled = torch.multinomial(output.view(-1, self.range_size), 1).view(x.shape)
        accuracy_per_position = (sampled == x).float().mean(dim=0)
        total_accuracy = (sampled == x).float().mean()

        return {
            "mean_entropy": mean_entropy,
            "total_accuracy": total_accuracy,
            "entropy_per_position": mean_entropy_per_position,
            "accuracy_per_position": accuracy_per_position,
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch[0]
        output = self(x)
        loss = self._compute_loss(batch)

        metrics = self._compute_metrics(x, output)

        self.log("train_loss", loss)
        self.log("train_mean_entropy", metrics["mean_entropy"])
        self.log("train_accuracy", metrics["total_accuracy"])

        for i, (ent, acc) in enumerate(
            zip(metrics["entropy_per_position"], metrics["accuracy_per_position"])
        ):
            self.log(f"train_entropy_pos_{i}", ent)
            self.log(f"train_accuracy_pos_{i}", acc)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x = batch[0]
        output = self(x)
        loss = self._compute_loss(batch)

        metrics = self._compute_metrics(x, output)

        self.log("val_loss", loss)
        self.log("val_mean_entropy", metrics["mean_entropy"])
        self.log("val_accuracy", metrics["total_accuracy"])

        for i, (ent, acc) in enumerate(
            zip(metrics["entropy_per_position"], metrics["accuracy_per_position"])
        ):
            self.log(f"val_entropy_pos_{i}", ent)
            self.log(f"val_accuracy_pos_{i}", acc)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


torch.autograd.set_detect_anomaly(True)

TUPLE_SIZE = 3
RANGE_SIZE = 32
EMBEDDING_DIM = 64
BATCH_SIZE = 2048
NUM_EPOCHS = 100

datamodule = TupleDataModule(
    tuple_size=TUPLE_SIZE, range_size=RANGE_SIZE, batch_size=BATCH_SIZE
)

model = TupleAutoencoder(TUPLE_SIZE, RANGE_SIZE, EMBEDDING_DIM)

run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb_logger = WandbLogger(run_name, project="tuple-autoencoder")

log_dir = Path("logs") / run_name
plot_embeddings_callback = PlotEmbeddingsCallback(log_dir)
plot_weights_callback = PlotLinearCallback(model.linear, "linear", log_dir)

trainer = L.Trainer(
    max_epochs=NUM_EPOCHS,
    accelerator="auto",
    devices=1,
    logger=wandb_logger,
    callbacks=[plot_embeddings_callback, plot_weights_callback],
)
trainer.fit(model, datamodule)
wandb_logger.finalize("success")
