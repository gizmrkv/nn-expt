from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from nn_expt.callback import WeightBiasHeatmap
from nn_expt.data.sequence import SequenceIdentityDataModule
from nn_expt.system.seq2seq import LinearSeq2SeqSystem
from nn_expt.utils import get_run_name, init_weights


def main():
    wandb.init()
    config = wandb.config

    datamodule = SequenceIdentityDataModule(
        config.max_length,
        config.vocab_size,
        config.batch_size,
        config.sample_size,
        train_ratio=config.train_ratio,
    )
    system = LinearSeq2SeqSystem(
        config.max_length,
        config.vocab_size,
        embedding_dim=config.embedding_dim,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    system.apply(init_weights)

    run_name = get_run_name()
    log_dir = Path("logs") / run_name
    wandb_logger = WandbLogger(run_name, project="seq2seq-embedding")

    def get_embedding(system: LinearSeq2SeqSystem) -> torch.Tensor:
        return system.embedding.weight

    def get_linear_weight(system: LinearSeq2SeqSystem) -> torch.Tensor:
        return system.linear.weight

    def get_linear_bias(system: LinearSeq2SeqSystem) -> torch.Tensor:
        return system.linear.bias

    embedding_heatmap = WeightBiasHeatmap(
        get_embedding,  # type: ignore
        save_dir=log_dir,
        name="embedding",
    )
    linear_heatmap = WeightBiasHeatmap(
        get_linear_weight,  # type: ignore
        get_linear_bias,  # type: ignore
        save_dir=log_dir,
        name="linear",
    )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        callbacks=[embedding_heatmap, linear_heatmap],
    )
    trainer.fit(system, datamodule)


if __name__ == "__main__":
    main()
