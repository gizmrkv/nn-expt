import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List

import lightning as L
import optuna
from lightning.pytorch.loggers import TensorBoardLogger

from nn_expt.callback import ObjectiveTracker, PositionalProbabilityHeatmap
from nn_expt.data.seq2seq import Seq2SeqDataModule
from nn_expt.optimize import optimize
from nn_expt.system.seq2seq2seq.mlp_mlp import MLPMLPSeq2Seq2SeqSystem


def objective(
    vocab_size: int,
    seq_len: int,
    z_vocab_size: int,
    z_seq_len: int,
    *,
    system_kwargs: Dict[str, Any],
    max_epochs: int,
    batch_size: int,
    train_ratio: float,
    num_workers: int,
    num_repeats: int,
    frame_every_n_epochs: int | None = None,
    seed: int | None = None,
    dir: str | Path = "./out/seq2seq2seq_mlp_mlp",
    trial: optuna.Trial | None = None,
) -> float:
    L.seed_everything(seed)

    hparams = locals()
    hparams.pop("trial")

    dir = Path(dir)

    name = (
        f"{trial.number:08d}"
        if trial is not None
        else datetime.datetime.now().isoformat()
    )
    tb_logger = TensorBoardLogger(dir / "tensorboard", name=name)
    tb_logger.log_hyperparams(hparams)

    system = MLPMLPSeq2Seq2SeqSystem(
        vocab_size, seq_len, z_vocab_size, z_seq_len, **system_kwargs
    )

    objective_tracker = ObjectiveTracker("val/receiver_loss")
    callbacks: List[L.Callback] = [objective_tracker]
    if frame_every_n_epochs is not None:
        callbacks.extend(
            [
                PositionalProbabilityHeatmap(
                    system.sender,
                    vocab_size,
                    seq_len,
                    vocab_size,
                    seq_len,
                    name="positional_probability_sender",
                    save_dir=dir / "positional_probability_sender",
                    frame_every_n_epochs=frame_every_n_epochs,
                ),
                PositionalProbabilityHeatmap(
                    system.receiver,
                    vocab_size,
                    seq_len,
                    vocab_size,
                    seq_len,
                    name="positional_probability_receiver",
                    save_dir=dir / "positional_probability_receiver",
                    frame_every_n_epochs=frame_every_n_epochs,
                ),
            ]
        )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=tb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )
    data = Seq2SeqDataModule(
        seq_len,
        vocab_size,
        batch_size,
        train_ratio=train_ratio,
        num_workers=num_workers,
        num_repeats=num_repeats,
    )

    trainer.fit(system, data)

    tb_logger.finalize("success")

    return objective_tracker.best_value or float("nan")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default=None)
    args = parser.parse_args()

    optimize(objective, args.config)
