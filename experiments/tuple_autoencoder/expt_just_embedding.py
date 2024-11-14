import datetime
from pathlib import Path
from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger

from nn_expt.callback import WeightBiasHeatmap
from nn_expt.data import TupleReconstructDataModule, TupleSortDataModule
from nn_expt.system import Tuple2TupleJustEmbeddingSystem

TUPLE_SIZE = 1
RANGE_SIZE = 50
BATCH_SIZE = 2048
NUM_EPOCHS = 50
SAMPLE_SIZE = 10000
LR = 0.01

# datamodule = TupleDataModule(
datamodule = TupleReconstructDataModule(
    tuple_size=TUPLE_SIZE,
    range_size=RANGE_SIZE,
    batch_size=BATCH_SIZE,
    sample_size=SAMPLE_SIZE,
    train_ratio=1,
)

system = Tuple2TupleJustEmbeddingSystem(TUPLE_SIZE, RANGE_SIZE, lr=LR)

run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb_logger = WandbLogger(run_name, project="tuple-autoencoder")


def get_embedding(system: Tuple2TupleJustEmbeddingSystem) -> torch.Tensor:
    return F.softmax(system.embedding.weight, dim=1)


log_dir = Path("logs") / run_name
embedding_heatmap = WeightBiasHeatmap(
    get_embedding,  # type: ignore
    save_dir=log_dir,
    name="embedding",
)
trainer = L.Trainer(
    max_epochs=NUM_EPOCHS,
    logger=wandb_logger,
    callbacks=[embedding_heatmap],
)
trainer.fit(system, datamodule)
wandb_logger.finalize("success")
