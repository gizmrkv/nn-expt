from pathlib import Path
from typing import List

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

from nn_expt.callback import PositionalProbabilityHeatmap
from nn_expt.data.sequence import SequenceIdentityDataModule
from nn_expt.nn import Seq2SeqLinear, Seq2SeqRNNDecoder, Seq2SeqRNNEncoder
from nn_expt.system.seq2seq import Seq2SeqSystem
from nn_expt.utils import get_run_name


def run_seq2seq():
    wandb.init(project="seq2seq")
    config = wandb.config

    L.seed_everything(config.seed)

    if config.model_type == "linear":
        model = Seq2SeqLinear(
            config.vocab_size,
            config.max_length,
            config.vocab_size,
            config.max_length,
            embedding_dim=config.embedding_dim,
            one_hot=config.one_hot,
            noisy=config.noisy,
        )
    elif config.model_type == "rnn_encoder":
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
    elif config.model_type == "rnn_decoder":
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
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    system = Seq2SeqSystem(
        model,
        optimizer=torch.optim.Adam(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        ),
    )

    run_name = get_run_name()
    log_dir = Path("logs") / run_name
    wandb_logger = WandbLogger(run_name)

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
                )
            ]
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
