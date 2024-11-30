import itertools
from pathlib import Path
from typing import Callable, List

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
        sender: nn.Module,
        vocab_size: int,
        max_length: int,
        z_vocab_size: int,
        z_max_length: int,
        *,
        save_dir: str | Path,
        name: str,
        fps: int = 20,
        frame_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.sender = sender
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.z_vocab_size = z_vocab_size
        self.z_max_length = z_max_length
        self.save_dir = Path(save_dir) / name
        self.name = name
        self.fps = fps
        self.frame_every_n_epochs = frame_every_n_epochs

        self.data = torch.tensor(
            list(itertools.product(range(vocab_size), repeat=max_length)),
            dtype=torch.long,
        )

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        epoch = trainer.current_epoch
        if epoch % self.frame_every_n_epochs != 0:
            return

        with torch.no_grad():
            output = self.sender(self.data.to(pl_module.device))
            output = output.softmax(dim=-1)
            output = output.view(-1, self.z_vocab_size * self.z_max_length)
            output = torch.concat([self.data.cpu(), output.cpu()], dim=1)

        data = pl.from_numpy(
            output.numpy(),
            schema=[f"i_{i}" for i in range(self.max_length)]
            + [
                f"o_{i}_{j}"
                for i in range(self.z_vocab_size)
                for j in range(self.z_max_length)
            ],
        ).with_columns(cs.starts_with("i").cast(pl.Int32))
        data = pl.concat(
            [
                data.select(f"i_{i}", cs.starts_with("o"))
                .group_by(f"i_{i}", maintain_order=True)
                .mean()
                .drop(f"i_{i}")
                for i in range(self.max_length)
            ]
        )

        plt.figure(figsize=(10, 8))
        sns.heatmap(data, cmap="viridis", vmin=0.0, vmax=1.0)
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
