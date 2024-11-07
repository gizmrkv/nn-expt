import datetime
from pathlib import Path
from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from callback import PlotLinearCallback
from data_module import TupleDataModule
from lightning.pytorch.loggers import WandbLogger
from torch.distributions import Categorical


class TupleLinearAutoencoder(L.LightningModule):
    def __init__(
        self,
        tuple_size: int,
        range_size: int,
        embedding_dim: int,
        *,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tuple_size = tuple_size
        self.range_size = range_size
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.weight_decay = weight_decay

        self.embedding = nn.Embedding(range_size, embedding_dim)
        self.linear = nn.Linear(tuple_size * embedding_dim, tuple_size * range_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.embedding(x)
        emb = emb.view(-1, self.tuple_size * self.embedding_dim)
        logits: torch.Tensor = self.linear(emb)
        logits = logits.view(-1, self.tuple_size, self.range_size)
        return logits

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        input, target = batch
        logits = self(input)
        loss = (
            F.cross_entropy(
                logits.view(-1, self.range_size),
                target.view(-1),
                reduction="none",
            )
            .view(-1, self.tuple_size)
            .mean(-1)
        )

        distr = Categorical(logits=logits)
        action = logits.argmax(dim=-1)
        acc_mean = (action == target).float().mean()
        entropy = distr.entropy().mean()
        self.log("acc_mean", acc_mean, prog_bar=True)
        self.log("entropy", entropy)

        for i in range(self.tuple_size):
            acc_i = (action[:, i] == target[:, i]).float().mean()
            ent_i = distr.entropy()[:, i].mean()
            self.log(f"acc_{i}", acc_i)
            self.log(f"ent_{i}", ent_i)

        return loss.mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


TUPLE_SIZE = 3
RANGE_SIZE = 100
EMBEDDING_DIM = 64
BATCH_SIZE = 2048
NUM_EPOCHS = 10
SAMPLE_SIZE = 1000000

datamodule = TupleDataModule(
    tuple_size=TUPLE_SIZE,
    range_size=RANGE_SIZE,
    batch_size=BATCH_SIZE,
    sample_size=SAMPLE_SIZE,
)

model = TupleLinearAutoencoder(TUPLE_SIZE, RANGE_SIZE, EMBEDDING_DIM)

run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb_logger = WandbLogger(run_name, project="tuple-autoencoder")

log_dir = Path("logs") / run_name
plot_weights_callback = PlotLinearCallback(model.linear, "linear", log_dir)

trainer = L.Trainer(
    max_epochs=NUM_EPOCHS,
    logger=wandb_logger,
    callbacks=[plot_weights_callback],
)
trainer.fit(model, datamodule)
wandb_logger.finalize("success")
