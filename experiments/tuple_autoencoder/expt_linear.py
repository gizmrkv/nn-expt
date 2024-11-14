import datetime
from pathlib import Path
from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger

from nn_expt.callback import WeightBiasHeatmap
from nn_expt.data import TupleReconstructDataModule, TupleSortDataModule
from nn_expt.system import Tuple2TupleLinearSystem

TUPLE_SIZE = 1
RANGE_SIZE = 100
EMBEDDING_DIM = 8
BATCH_SIZE = 2048
NUM_EPOCHS = 50
SAMPLE_SIZE = 10000

# datamodule = TupleDataModule(
datamodule = TupleReconstructDataModule(
    tuple_size=TUPLE_SIZE,
    range_size=RANGE_SIZE,
    batch_size=BATCH_SIZE,
    sample_size=SAMPLE_SIZE,
    train_ratio=1,
)

system = Tuple2TupleLinearSystem(TUPLE_SIZE, RANGE_SIZE, EMBEDDING_DIM)

run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb_logger = WandbLogger(run_name, project="tuple-autoencoder")


def get_embedding(system: Tuple2TupleLinearSystem) -> torch.Tensor:
    return system.embedding.weight


def get_weight(system: Tuple2TupleLinearSystem) -> torch.Tensor:
    return system.linear.weight


def get_bias(system: Tuple2TupleLinearSystem) -> torch.Tensor:
    return system.linear.bias


log_dir = Path("logs") / run_name
embedding_heatmap = WeightBiasHeatmap(
    get_embedding,  # type: ignore
    save_dir=log_dir,
    name="embedding",
)
weight_bias_heatmap = WeightBiasHeatmap(
    get_weight,  # type: ignore
    get_bias,  # type: ignore
    save_dir=log_dir,
    name="weight_bias",
)

trainer = L.Trainer(
    max_epochs=NUM_EPOCHS,
    logger=wandb_logger,
    callbacks=[embedding_heatmap, weight_bias_heatmap],
)
trainer.fit(system, datamodule)
wandb_logger.finalize("success")
