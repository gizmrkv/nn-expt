import argparse
from pathlib import Path
from typing import List

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import wandb
from nn_expt.callback import (
    PositionalProbabilityHeatmap,
    embedding_weight_heatmap,
    linear_weight_bias_heatmap,
    rnn_weight_bias_heatmaps,
)
from nn_expt.data.sequence import SequenceIdentityDataModule
from nn_expt.nn import Seq2SeqRNNDecoder, Seq2SeqRNNEncoder
from nn_expt.system.seq2seq import Seq2SeqSystem
from nn_expt.utils import get_run_name


def main():
    wandb.init()
    config = wandb.config

    L.seed_everything(config.seed)

    if config.rnn_mode == "encoder":
        model = Seq2SeqRNNEncoder(
            config.vocab_size,
            config.vocab_size,
            config.max_length,
            embedding_dim=config.embedding_dim,
            one_hot=config.one_hot,
            hidden_size=config.hidden_size,
            rnn_type=config.rnn_type,
            num_layers=config.num_layers,
            bias=config.bias,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
        )
    else:
        model = Seq2SeqRNNDecoder(
            config.vocab_size,
            config.max_length,
            config.vocab_size,
            config.max_length,
            embedding_dim=config.embedding_dim,
            one_hot=config.one_hot,
            hidden_size=config.hidden_size,
            rnn_type=config.rnn_type,
            num_layers=config.num_layers,
            bias=config.bias,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
        )
    system = Seq2SeqSystem(
        model,
        optimizer=torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        ),
    )

    run_name = get_run_name()
    log_dir = Path("logs") / run_name
    wandb_logger = WandbLogger(run_name, project="seq2seq")

    callbacks: List[L.Callback] = [
        EarlyStopping("val/loss", mode="min", patience=config.patience)
    ]
    if config.frame_every_n_epochs > 0:
        callbacks.extend(
            [
                PositionalProbabilityHeatmap(
                    model,
                    config.vocab_size,
                    config.max_length,
                    config.vocab_size,
                    config.max_length,
                    save_dir=log_dir,
                    name="positional_probability",
                    frame_every_n_epochs=config.frame_every_n_epochs,
                ),
                embedding_weight_heatmap(
                    model.embedding,
                    save_dir=log_dir,
                    name="embedding",
                    frame_every_n_epochs=config.frame_every_n_epochs,
                ),
                linear_weight_bias_heatmap(
                    model.linear,
                    save_dir=log_dir,
                    name="encoder",
                    frame_every_n_epochs=config.frame_every_n_epochs,
                ),
                *rnn_weight_bias_heatmaps(
                    model.rnn,
                    save_dir=log_dir,
                    name="decoder",
                    frame_every_n_epochs=config.frame_every_n_epochs,
                ),
            ],
        )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )
    datamodule = SequenceIdentityDataModule(
        config.max_length,
        config.vocab_size,
        config.batch_size,
        train_ratio=config.train_ratio,
        num_workers=config.num_workers,
        n_repeats=config.n_repeats,
    )
    trainer.fit(system, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None)
    args = parser.parse_args()

    sweep_id: str | None = args.sweep_id

    if sweep_id is None:
        pwd = Path(__file__).parent
        path = pwd / "config/rnn_1.yml"
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep=config, project="seq2seq")

    wandb.agent(sweep_id, function=main)
