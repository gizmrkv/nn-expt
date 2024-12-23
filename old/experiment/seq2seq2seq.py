from pathlib import Path
from typing import Any, Dict, List

import lightning as L
import optuna
import torch
from lightning.pytorch.loggers import TensorBoardLogger

from nn_expt.callback import ObjectiveTracker, PositionalProbabilityHeatmap
from nn_expt.data.sequence import Seq2SeqDataModule
from nn_expt.nn import Seq2SeqLinear, Seq2SeqRNNDecoder, Seq2SeqRNNEncoder
from nn_expt.system import Seq2Seq2SeqSystem
from nn_expt.utils import get_run_name


def run_seq2seq2seq(
    vocab_size: int,
    max_length: int,
    z_vocab_size: int,
    z_max_length: int,
    *,
    sender_model_type: str,
    sender_model_params: Dict[str, Any],
    sender_optimizer_params: Dict[str, Any],
    sender_entropy_weight: float,
    receiver_model_type: str,
    receiver_model_params: Dict[str, Any],
    receiver_optimizer_params: Dict[str, Any],
    max_epochs: int,
    batch_size: int,
    train_ratio: float,
    num_workers: int,
    n_repeats: int,
    seed: int,
    frame_every_n_epochs: int = 1,
    trial: optuna.Trial,
):
    L.seed_everything(seed)

    hparams = locals()

    run_name = get_run_name()
    log_dir = Path("out/logs") / run_name

    tb_logger = TensorBoardLogger("out/logs", name=f"{trial.number:08}")
    tb_logger.log_hyperparams(hparams)

    if sender_model_type == "linear":
        sender = Seq2SeqLinear(
            vocab_size, max_length, z_vocab_size, z_max_length, **sender_model_params
        )
    elif sender_model_type == "rnn_encoder":
        sender = Seq2SeqRNNEncoder(
            vocab_size, z_vocab_size, z_max_length, **sender_model_params
        )
    elif sender_model_type == "rnn_decoder":
        sender = Seq2SeqRNNDecoder(
            vocab_size, max_length, z_vocab_size, z_max_length, **sender_model_params
        )
    else:
        raise ValueError(f"Unknown sender_type: {sender_model_type}")

    if receiver_model_type == "linear":
        receiver = Seq2SeqLinear(
            z_vocab_size, z_max_length, vocab_size, max_length, **receiver_model_params
        )
    elif receiver_model_type == "rnn_encoder":
        receiver = Seq2SeqRNNEncoder(
            z_vocab_size, vocab_size, max_length, **receiver_model_params
        )
    elif receiver_model_type == "rnn_decoder":
        receiver = Seq2SeqRNNDecoder(
            z_vocab_size, z_max_length, vocab_size, max_length, **receiver_model_params
        )
    else:
        raise ValueError(f"Unknown receiver_type: {receiver_model_type}")

    system = Seq2Seq2SeqSystem(
        sender,
        receiver,
        optimizer=torch.optim.SGD(
            [
                {"params": sender.parameters(), **sender_optimizer_params},
                {"params": receiver.parameters(), **receiver_optimizer_params},
            ]
        ),
        sender_entropy_weight=sender_entropy_weight,
    )

    objective_tracker = ObjectiveTracker("val/receiver_loss")
    callbacks: List[L.Callback] = [objective_tracker]
    if frame_every_n_epochs > 0:
        callbacks.extend(
            [
                PositionalProbabilityHeatmap(
                    sender,
                    vocab_size,
                    max_length,
                    z_vocab_size,
                    z_max_length,
                    save_dir=log_dir,
                    name="sender_positional_probability",
                    frame_every_n_epochs=frame_every_n_epochs,
                ),
                PositionalProbabilityHeatmap(
                    receiver,
                    z_vocab_size,
                    z_max_length,
                    vocab_size,
                    max_length,
                    save_dir=log_dir,
                    name="receiver_positional_probability",
                    frame_every_n_epochs=frame_every_n_epochs,
                ),
            ]
        )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=1,
        logger=tb_logger,
        callbacks=callbacks,
    )
    datamodule = Seq2SeqDataModule(
        max_length,
        vocab_size,
        batch_size,
        train_ratio=train_ratio,
        num_workers=num_workers,
        num_repeats=n_repeats,
    )

    trainer.fit(system, datamodule)

    tb_logger.finalize("success")

    return objective_tracker.best_value or float("nan")
