import datetime
from pathlib import Path
from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
from callback import PlotLinearCallback
from data_module import TupleDataModule
from lightning.pytorch.loggers import WandbLogger
from torch.distributions import Categorical
from tuple_autoencoder import TupleAutoencoder


class LinearReinforceTupleAutoencoder(TupleAutoencoder):
    def __init__(
        self,
        tuple_size: int,
        range_size: int,
        embedding_dim: int,
        *,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            tuple_size=tuple_size,
            range_size=range_size,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(range_size, embedding_dim)
        self.linear = nn.Linear(tuple_size * embedding_dim, tuple_size * range_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        emb: torch.Tensor = self.embedding(x)
        emb = emb.view(-1, self.tuple_size * self.embedding_dim)
        logits: torch.Tensor = self.linear(emb)
        logits = logits.view(-1, self.tuple_size, self.range_size)
        return logits, None

    def calc_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        distr = Categorical(logits=logits)
        action = distr.sample()
        log_prob = distr.log_prob(action).sum(dim=-1)
        reward = (action == target).float().mean(dim=-1)
        loss = -(reward - reward.mean()) / (reward.std() + 1e-8) * log_prob
        return loss.mean()


TUPLE_SIZE = 3
RANGE_SIZE = 50
EMBEDDING_DIM = 64
BATCH_SIZE = 2048
NUM_EPOCHS = 100
SAMPLE_SIZE = 10000

datamodule = TupleDataModule(
    tuple_size=TUPLE_SIZE,
    range_size=RANGE_SIZE,
    batch_size=BATCH_SIZE,
    sample_size=SAMPLE_SIZE,
)

model = LinearReinforceTupleAutoencoder(TUPLE_SIZE, RANGE_SIZE, EMBEDDING_DIM)

run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb_logger = WandbLogger(run_name, project="tuple-autoencoder")

log_dir = Path("logs") / run_name
plot_weights_callback = PlotLinearCallback(model.linear, "linear", log_dir)

trainer = L.Trainer(
    max_epochs=NUM_EPOCHS,
    log_every_n_steps=1,
    logger=wandb_logger,
    callbacks=[plot_weights_callback],
)
trainer.fit(model, datamodule)
wandb_logger.finalize("success")
