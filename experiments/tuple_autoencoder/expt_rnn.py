import datetime
from pathlib import Path
from typing import List, Literal, Tuple

import lightning as L
import torch
import torch.nn as nn
from callback import PlotLinearCallback
from data_module import TupleDataModule
from lightning.pytorch.loggers import WandbLogger
from torch.distributions import Categorical
from tuple_autoencoder import TupleAutoencoder


class SequentialRNNTupleAutoencoder(TupleAutoencoder):
    def __init__(
        self,
        tuple_size: int,
        range_size: int,
        embedding_dim: int,
        hidden_size: int,
        rnn_type: Literal["RNN", "LSTM", "GRU"],
        num_layers: int = 1,
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
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        self.embedding = nn.Embedding(range_size, embedding_dim)
        self.hidden_linears = nn.ModuleList(
            nn.Linear(tuple_size * embedding_dim, hidden_size)
            for _ in range(num_layers)
        )
        self.start_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        self.output_linear = nn.Linear(hidden_size, range_size)

        rnn_dict = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
        }
        self.rnn = rnn_dict[rnn_type](
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x).view(-1, self.tuple_size * self.embedding_dim)
        h = torch.stack([linear(x) for linear in self.hidden_linears])
        if isinstance(self.rnn, nn.LSTM):
            c = torch.zeros_like(h)

        i = self.start_embedding.repeat(x.size(0), 1)

        symbol_list: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        for _ in range(self.tuple_size):
            i = i.unsqueeze(1)
            if isinstance(self.rnn, nn.LSTM):
                y, (h, c) = self.rnn(i, (h, c))  # type: ignore
            else:
                y, h = self.rnn(i, h)

            logits = self.output_linear(y.squeeze(1))
            distr = Categorical(logits=logits)

            if self.training:
                x = distr.sample()
            else:
                x = logits.argmax(dim=-1)

            i = self.embedding(x)
            symbol_list.append(x)
            logits_list.append(logits)

        symbols = torch.stack(symbol_list, dim=1)
        logits = torch.stack(logits_list, dim=1)
        return logits, symbols


TUPLE_SIZE = 3
RANGE_SIZE = 50
EMBEDDING_DIM = 32
HIDDEN_SIZE = 64
RNN_TYPE = "RNN"
BATCH_SIZE = 2048
NUM_EPOCHS = 100
SAMPLE_SIZE = 10000

datamodule = TupleDataModule(
    tuple_size=TUPLE_SIZE,
    range_size=RANGE_SIZE,
    batch_size=BATCH_SIZE,
    sample_size=SAMPLE_SIZE,
    train_ratio=0.1,
)

model = SequentialRNNTupleAutoencoder(
    TUPLE_SIZE, RANGE_SIZE, EMBEDDING_DIM, HIDDEN_SIZE, RNN_TYPE
)

run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb_logger = WandbLogger(run_name, project="tuple-autoencoder")

log_dir = Path("logs") / run_name

trainer = L.Trainer(max_epochs=NUM_EPOCHS, log_every_n_steps=1, logger=wandb_logger)
trainer.fit(model, datamodule)
wandb_logger.finalize("success")
