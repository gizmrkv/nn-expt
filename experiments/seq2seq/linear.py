from pathlib import Path

import lightning as L
from lightning.pytorch.loggers import WandbLogger

import wandb
from nn_expt.callback import WeightBiasHeatmap
from nn_expt.data.sequence import SequenceIdentityDataModule
from nn_expt.nn import Seq2SeqLinear
from nn_expt.system.seq2seq import Seq2SeqSystem
from nn_expt.utils import get_run_name


def main():
    wandb.init()
    config = wandb.config

    L.seed_everything(config.seed)

    model = Seq2SeqLinear(
        config.vocab_size,
        config.max_length,
        config.vocab_size,
        config.max_length,
        embedding_dim=config.embedding_dim,
        one_hot=config.one_hot,
        noisy=config.noisy,
    )
    system = Seq2SeqSystem(
        model,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    run_name = get_run_name()
    log_dir = Path("logs") / run_name
    wandb_logger = WandbLogger(run_name, project="seq2seq")

    callbacks = []
    callbacks.append(
        WeightBiasHeatmap(
            lambda system: system.model.embedding.weight,
            save_dir=log_dir,
            name="embedding",
            frame_every_n_epochs=config.frame_every_n_epochs,
        )
    )
    callbacks.append(
        WeightBiasHeatmap(
            lambda system: system.model.linear.weight,
            lambda system: system.model.linear.bias,
            save_dir=log_dir,
            name="linear",
            frame_every_n_epochs=config.frame_every_n_epochs,
        )
    )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
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
