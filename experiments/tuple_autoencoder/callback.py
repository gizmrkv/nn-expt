from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from moviepy.editor import ImageSequenceClip

import wandb


class PlotEmbeddingsCallback(Callback):
    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        embeddings = pl_module.embedding.weight.detach().cpu().numpy()

        plt.figure(figsize=(10, 10))
        plt.scatter(embeddings[:, 0], embeddings[:, 1])

        for i in range(len(embeddings)):
            plt.annotate(str(i), (embeddings[i, 0], embeddings[i, 1]))

        plt.title(f"Embedding Space at Epoch {trainer.current_epoch}")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")

        zero_pad = len(str(trainer.max_epochs))
        filename = str(
            self.log_dir / f"embeddings_epoch_{trainer.current_epoch:0{zero_pad}}.png"
        )

        plt.savefig(filename)
        plt.close()

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_image(key="embeddings", images=[wandb.Image(filename)])

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        image_files = sorted(list(self.log_dir.glob("embeddings_epoch_*.png")))

        clip = ImageSequenceClip([str(img) for img in image_files], fps=5)

        clip.write_videofile(
            str(self.log_dir / "embedding_evolution.mp4"),
            fps=5,
            codec="libx264",
            audio=False,
        )

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.experiment.log(
                {
                    "embedding_evolution": wandb.Video(
                        str(self.log_dir / "embedding_evolution.mp4"),
                        fps=5,
                        format="mp4",
                    )
                }
            )


class PlotLinearCallback(Callback):
    def __init__(self, linear: nn.Linear, name: str, log_dir: Path):
        super().__init__()
        self.linear = linear
        self.name = name
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        weights = self.linear.weight.detach().cpu().numpy()
        bias = self.linear.bias.detach().cpu().numpy()
        weights = np.concatenate([weights, bias[:, None]], axis=1)

        max_abs_val = np.abs(weights).max()
        plt.figure(figsize=(12, 8))
        plt.imshow(
            weights, cmap="RdBu", aspect="auto", vmin=-max_abs_val, vmax=max_abs_val
        )
        plt.colorbar(label="Weight Value")

        plt.title(f"Linear Layer Weights at Epoch {trainer.current_epoch}")
        plt.xlabel("Input Features")
        plt.ylabel("Output Features")

        zero_pad = len(str(trainer.max_epochs))
        filename = self.log_dir / f"{self.name}_{trainer.current_epoch:0{zero_pad}}.png"
        plt.savefig(filename)
        plt.close()

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_image(key=self.name, images=[filename.as_posix()])

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        image_files = sorted(list(self.log_dir.glob(f"{self.name}_*.png")))

        clip = ImageSequenceClip([str(img) for img in image_files], fps=5)
        filename = self.log_dir / f"{self.name}_evolution.mp4"
        clip.write_videofile(filename.as_posix(), fps=5, codec="libx264", audio=False)

        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_video(
                key=f"{self.name}_evolution", videos=[filename.as_posix()]
            )
