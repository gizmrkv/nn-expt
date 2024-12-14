from pathlib import Path
from typing import Any, Dict, List, Literal

import lightning as L
import torch
from lightning.pytorch.loggers import WandbLogger

import wandb
from nn_expt.callback import ObjectiveTracker, PositionalProbabilityHeatmap
from nn_expt.data.sequence import SequenceIdentityDataModule
from nn_expt.nn import Seq2SeqLinear, Seq2SeqRNNDecoder, Seq2SeqRNNEncoder
from nn_expt.system.seq2seq import Seq2SeqSystem
from nn_expt.utils import get_run_name


def run_seq2seq(
    vocab_size: int,
    max_length: int,
    *,
    model_type: Literal["linear", "rnn_encoder", "rnn_decoder"],
    model_params: Dict[str, Any],
    optimizer_params: Dict[str, Any],
    max_epochs: int,
    batch_size: int,
    train_ratio: float,
    num_workers: int,
    n_repeats: int,
    seed: int,
    frame_every_n_epochs: int = 1,
) -> float:
    wandb.init(project="seq2seq", dir="./out")
    wandb.config.update(locals())

    L.seed_everything(seed)

    if model_type == "linear":
        model = Seq2SeqLinear(
            vocab_size, max_length, vocab_size, max_length, **model_params
        )
    elif model_type == "rnn_encoder":
        model = Seq2SeqRNNEncoder(vocab_size, vocab_size, max_length, **model_params)
    elif model_type == "rnn_decoder":
        model = Seq2SeqRNNDecoder(
            vocab_size, max_length, vocab_size, max_length, **model_params
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    system = Seq2SeqSystem(
        model,
        optimizer=torch.optim.SGD(model.parameters(), **optimizer_params),
    )

    run_name = get_run_name()
    log_dir = Path("logs") / run_name
    wandb_logger = WandbLogger(run_name, save_dir="./out")

    objective_tracker = ObjectiveTracker("val/loss")
    callbacks: List[L.Callback] = [objective_tracker]
    if frame_every_n_epochs > 0:
        callbacks.extend(
            [
                PositionalProbabilityHeatmap(
                    model,
                    vocab_size,
                    max_length,
                    vocab_size,
                    max_length,
                    save_dir=log_dir,
                    name="positional_probability",
                    frame_every_n_epochs=frame_every_n_epochs,
                )
            ]
        )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=wandb_logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )
    datamodule = SequenceIdentityDataModule(
        max_length,
        vocab_size,
        batch_size,
        train_ratio=train_ratio,
        num_workers=num_workers,
        n_repeats=n_repeats,
    )
    trainer.fit(system, datamodule)

    wandb_logger.finalize("success")
    wandb.finish()

    return objective_tracker.best_value or float("nan")
