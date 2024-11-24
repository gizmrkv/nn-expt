from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import wandb
from nn_expt.callback import WeightBiasHeatmap
from nn_expt.data.sequence import SequenceIdentityDataModule
from nn_expt.nn import Seq2SeqLinear
from nn_expt.system import Seq2Seq2SeqSystem
from nn_expt.utils import get_run_name


def main():
    wandb.init()
    config = wandb.config

    L.seed_everything(config.seed)

    encoder = Seq2SeqLinear(
        config.vocab_size,
        config.max_length,
        config.z_vocab_size,
        config.z_max_length,
        embedding_dim=config.encode_embedding_dim,
        one_hot=config.encode_one_hot,
        noisy=config.noisy,
    )
    decoder = Seq2SeqLinear(
        config.z_vocab_size,
        config.z_max_length,
        config.vocab_size,
        config.max_length,
        embedding_dim=config.decode_embedding_dim,
        one_hot=config.decode_one_hot,
    )
    system = Seq2Seq2SeqSystem(
        encoder,
        decoder,
        lr=config.lr,
        weight_decay=config.weight_decay,
        encode_entropy_weight=config.encode_entropy_weight,
    )

    run_name = get_run_name()
    log_dir = Path("logs") / run_name
    wandb_logger = WandbLogger(run_name, project="seq2seq2seq")

    callbacks = []
    callbacks.append(
        WeightBiasHeatmap(
            lambda system: system.encoder.embedding.weight,
            save_dir=log_dir,
            name="encode_embedding",
            frame_every_n_epochs=config.frame_every_n_epochs,
        )
    )
    callbacks.append(
        WeightBiasHeatmap(
            lambda system: system.decoder.embedding.weight,
            save_dir=log_dir,
            name="decode_embedding",
            frame_every_n_epochs=config.frame_every_n_epochs,
        )
    )
    callbacks.append(
        WeightBiasHeatmap(
            lambda system: system.encoder.linear.weight,
            lambda system: system.encoder.linear.bias,
            save_dir=log_dir,
            name="encode_linear",
            frame_every_n_epochs=config.frame_every_n_epochs,
        )
    )
    callbacks.append(
        WeightBiasHeatmap(
            lambda system: system.decoder.linear.weight,
            lambda system: system.decoder.linear.bias,
            save_dir=log_dir,
            name="decode_linear",
            frame_every_n_epochs=config.frame_every_n_epochs,
        )
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
