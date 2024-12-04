from pathlib import Path
from typing import List

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import wandb
from nn_expt.callback import PositionalProbabilityHeatmap
from nn_expt.data.sequence import SequenceIdentityDataModule
from nn_expt.nn import Seq2SeqLinear, Seq2SeqRNNDecoder, Seq2SeqRNNEncoder
from nn_expt.system import Seq2Seq2SeqSystem
from nn_expt.utils import get_run_name


def run_seq2seq2seq():
    wandb.init(project="seq2seq2seq")
    config = wandb.config

    L.seed_everything(config.seed)

    if config.sender_type == "linear":
        sender = Seq2SeqLinear(
            config.z_vocab_size,
            config.z_max_length,
            config.vocab_size,
            config.max_length,
            embedding_dim=config.sender_embedding_dim,
            one_hot=config.sender_one_hot,
            noisy=config.sender_noisy,
        )
    elif config.sender_type == "rnn_encoder":
        sender = Seq2SeqRNNEncoder(
            config.z_vocab_size,
            config.vocab_size,
            config.max_length,
            embedding_dim=config.sender_embedding_dim,
            one_hot=config.sender_one_hot,
            hidden_size=config.sender_hidden_size,
            rnn_type=config.sender_rnn_type,
            num_layers=config.sender_num_layers,
            bias=config.sender_bias,
            dropout=config.sender_dropout,
            bidirectional=config.sender_bidirectional,
        )
    elif config.sender_type == "rnn_decoder":
        sender = Seq2SeqRNNDecoder(
            config.z_vocab_size,
            config.z_max_length,
            config.vocab_size,
            config.max_length,
            embedding_dim=config.sender_embedding_dim,
            one_hot=config.sender_one_hot,
            hidden_size=config.sender_hidden_size,
            rnn_type=config.sender_rnn_type,
            num_layers=config.sender_num_layers,
            bias=config.sender_bias,
            dropout=config.sender_dropout,
        )
    else:
        raise ValueError(f"Unknown sender_type: {config.sender_type}")

    if config.receiver_type == "linear":
        receiver = Seq2SeqLinear(
            config.z_vocab_size,
            config.z_max_length,
            config.vocab_size,
            config.max_length,
            embedding_dim=config.receiver_embedding_dim,
            one_hot=config.receiver_one_hot,
        )
    elif config.receiver_type == "rnn_encoder":
        receiver = Seq2SeqRNNEncoder(
            config.z_vocab_size,
            config.vocab_size,
            config.max_length,
            embedding_dim=config.receiver_embedding_dim,
            one_hot=config.receiver_one_hot,
            hidden_size=config.receiver_hidden_size,
            rnn_type=config.receiver_rnn_type,
            num_layers=config.receiver_num_layers,
            bias=config.receiver_bias,
            dropout=config.receiver_dropout,
            bidirectional=config.receiver_bidirectional,
        )
    elif config.receiver_type == "rnn_decoder":
        receiver = Seq2SeqRNNDecoder(
            config.z_vocab_size,
            config.z_max_length,
            config.vocab_size,
            config.max_length,
            embedding_dim=config.receiver_embedding_dim,
            one_hot=config.receiver_one_hot,
            hidden_size=config.receiver_hidden_size,
            rnn_type=config.receiver_rnn_type,
            num_layers=config.receiver_num_layers,
            bias=config.receiver_bias,
            dropout=config.receiver_dropout,
        )
    else:
        raise ValueError(f"Unknown receiver_type: {config.receiver_type}")

    system = Seq2Seq2SeqSystem(
        sender,
        receiver,
        optimizer=torch.optim.Adam(
            [
                {
                    "params": receiver.parameters(),
                    "lr": config.sender_lr,
                    "weight_decay": config.sender_weight_decay,
                },
                {
                    "params": sender.parameters(),
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
                PositionalProbabilityHeatmap(
                    receiver,
                    config.z_vocab_size,
                    config.z_max_length,
                    config.vocab_size,
                    config.max_length,
                    save_dir=log_dir,
                    name="receiver_positional_probability",
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
