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
)
from nn_expt.data.sequence import SequenceIdentityDataModule
from nn_expt.nn import Seq2SeqLinear
from nn_expt.system import Seq2Seq2SeqSystem
from nn_expt.utils import get_run_name


def main():
    wandb.init(
        project="seq2seq2seq",
    )
    config = wandb.config

    L.seed_everything(config.seed)

    sender = Seq2SeqLinear(
        config.vocab_size,
        config.max_length,
        config.z_vocab_size,
        config.z_max_length,
        embedding_dim=config.sender_embedding_dim,
        one_hot=config.sender_one_hot,
        noisy=config.sender_noisy,
    )
    receiver = Seq2SeqLinear(
        config.z_vocab_size,
        config.z_max_length,
        config.vocab_size,
        config.max_length,
        embedding_dim=config.receiver_embedding_dim,
        one_hot=config.receiver_one_hot,
    )
    system = Seq2Seq2SeqSystem(
        sender,
        receiver,
        optimizer=torch.optim.Adam(
            [
                {
                    "params": sender.parameters(),
                    "lr": config.sender_lr,
                    "weight_decay": config.sender_weight_decay,
                },
                {
                    "params": receiver.parameters(),
                    "lr": config.receiver_lr,
                    "weight_decay": config.receiver_weight_decay,
                },
            ]
        ),
        sender_entropy_weight=config.sender_entropy_weight,
    )

    run_name = get_run_name()
    log_dir = Path("logs") / run_name
    wandb_logger = WandbLogger(run_name)

    callbacks: List[L.Callback] = [
        EarlyStopping("val/receiver_loss", mode="min", patience=config.patience)
    ]
    if config.frame_every_n_epochs > 0:
        callbacks.extend(
            [
                PositionalProbabilityHeatmap(
                    sender,
                    config.vocab_size,
                    config.max_length,
                    config.z_vocab_size,
                    config.z_max_length,
                    save_dir=log_dir,
                    name="sender_positional_probability",
                    frame_every_n_epochs=config.frame_every_n_epochs,
                ),
                embedding_weight_heatmap(
                    sender.embedding,
                    save_dir=log_dir,
                    name="sender_embedding",
                    frame_every_n_epochs=config.frame_every_n_epochs,
                ),
                linear_weight_bias_heatmap(
                    sender.linear,
                    save_dir=log_dir,
                    name="sender_linear",
                    frame_every_n_epochs=config.frame_every_n_epochs,
                ),
                embedding_weight_heatmap(
                    receiver.embedding,
                    save_dir=log_dir,
                    name="receiver_embedding",
                    frame_every_n_epochs=config.frame_every_n_epochs,
                ),
                linear_weight_bias_heatmap(
                    receiver.linear,
                    save_dir=log_dir,
                    name="receiver_linear",
                    frame_every_n_epochs=config.frame_every_n_epochs,
                ),
            ]
        )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        log_every_n_steps=1,
        logger=wandb_logger,
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
    parser.add_argument("--sweep_id", "-s", type=str, default=None)
    parser.add_argument("--config_path", "-c", type=str, default=None)

    args = parser.parse_args()
    sweep_id: str | None = args.sweep_id
    config_path: str | None = args.config_path

    if sweep_id is None and config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        sweep_id = wandb.sweep(config, project="seq2seq2seq")

    if sweep_id is None:
        raise ValueError("Wrong sweep_id or config_path")

    wandb.agent(sweep_id, function=main)
