from pathlib import Path
from typing import List

import lightning as L
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
        sender_lr=config.sender_lr,
        sender_weight_decay=config.sender_weight_decay,
        receiver_lr=config.receiver_lr,
        receiver_weight_decay=config.receiver_weight_decay,
        sender_entropy_weight=config.sender_entropy_weight,
    )

    run_name = get_run_name()
    log_dir = Path("logs") / run_name
    wandb_logger = WandbLogger(run_name)

    callbacks: List[L.Callback] = [
        EarlyStopping("val/acc_mean", mode="max", patience=config.patience)
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
    main()
