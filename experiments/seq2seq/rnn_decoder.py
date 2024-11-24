import lightning as L
from lightning.pytorch.loggers import WandbLogger

import wandb
from nn_expt.data.sequence import SequenceIdentityDataModule
from nn_expt.nn import Seq2SeqRNNDecoder
from nn_expt.system.seq2seq import Seq2SeqSystem
from nn_expt.utils import get_run_name


def main():
    wandb.init()
    config = wandb.config

    L.seed_everything(config.seed)

    model = Seq2SeqRNNDecoder(
        config.vocab_size,
        config.max_length,
        config.vocab_size,
        config.max_length,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        rnn_type=config.rnn_type,
        num_layers=config.num_layers,
        bias=config.bias,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    )
    system = Seq2SeqSystem(
        model,
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    run_name = get_run_name()
    wandb_logger = WandbLogger(run_name, project="seq2seq")

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        logger=wandb_logger,
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
