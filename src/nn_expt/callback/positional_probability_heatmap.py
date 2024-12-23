import itertools
from pathlib import Path
from typing import Callable

import lightning as L
import matplotlib.pyplot as plt
import polars as pl
import polars.selectors as cs
import seaborn as sns
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from moviepy.editor import ImageSequenceClip


class PositionalProbabilityHeatmap(L.Callback):
    def __init__(
        self,
        sender: nn.Module | Callable[[torch.Tensor], torch.Tensor],
        in_vocab_size: int,
        in_seq_len: int,
        out_vocab_size: int,
        out_seq_len: int,
        *,
        save_dir: str | Path,
        name: str,
        fps: int = 20,
        frame_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.sender = sender
        self.in_vocab_size = in_vocab_size
        self.in_seq_len = in_seq_len
        self.out_vocab_size = out_vocab_size
        self.out_seq_len = out_seq_len
        self.save_dir = Path(save_dir) / name
        self.name = name
        self.fps = fps
        self.frame_every_n_epochs = frame_every_n_epochs

        self.data = torch.tensor(
            list(itertools.product(range(in_vocab_size), repeat=in_seq_len)),
            dtype=torch.long,
        )

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        epoch = trainer.current_epoch
        if epoch % self.frame_every_n_epochs != 0:
            return

        with torch.no_grad():
            output = self.sender(self.data.to(pl_module.device))

            if isinstance(output, tuple):
                output, *_ = output

            output = output.softmax(dim=-1)
            output = output.view(-1, self.out_vocab_size * self.out_seq_len)
            output = torch.concat([self.data.cpu(), output.cpu()], dim=1)

        data = pl.from_numpy(
            output.numpy(),
            schema=[f"i_{i}" for i in range(self.in_seq_len)]
            + [
                f"o_{i}_{j}"
                for i in range(self.out_vocab_size)
                for j in range(self.out_seq_len)
            ],
        ).with_columns(cs.starts_with("i").cast(pl.Int32))
        data = pl.concat(
            [
                data.select(f"i_{i}", cs.starts_with("o"))
                .group_by(f"i_{i}", maintain_order=True)
                .agg(
                    cs.starts_with("o").mean().name.suffix("_mean"),
                    cs.starts_with("o").std().name.suffix("_std"),
                )
                .drop(f"i_{i}")
                for i in range(self.in_seq_len)
            ]
        )

        plt.figure(figsize=(14, 8))
        sns.heatmap(data, cmap="viridis", vmin=0.0)
        plt.title(f"Positional Agg Heatmap - Epoch {epoch}")

        image_path = self.save_dir / f"epoch_{epoch:04d}.png"
        plt.savefig(image_path)
        plt.close()

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        image_files = sorted(list(self.save_dir.glob("epoch_*.png")))
        images = [path.as_posix() for path in image_files]
        clip = ImageSequenceClip(images, fps=self.fps)
        video_path = self.save_dir / "positional_agg_evolution.mp4"
        clip.write_videofile(video_path.as_posix(), codec="libx264", audio=False)
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_video(
                key=f"{self.name}_evolution", videos=[video_path.as_posix()]
            )
