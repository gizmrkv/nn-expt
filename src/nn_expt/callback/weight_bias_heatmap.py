from pathlib import Path
from typing import Callable, List

import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from lightning.pytorch.loggers import WandbLogger
from moviepy.editor import ImageSequenceClip


class WeightBiasHeatmap(L.Callback):
    def __init__(
        self,
        weight_fn: Callable[[L.LightningModule], torch.Tensor],
        bias_fn: Callable[[L.LightningModule], torch.Tensor] | None = None,
        *,
        save_dir: str | Path,
        name: str,
        fps: int = 30,
        frame_every_n_epochs: int = 1,
    ):
        super().__init__()
        self.weight_fn = weight_fn
        self.bias_fn = bias_fn
        self.save_dir = Path(save_dir) / name
        self.name = name
        self.fps = fps
        self.frame_every_n_epochs = frame_every_n_epochs

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        epoch = trainer.current_epoch
        if epoch % self.frame_every_n_epochs != 0:
            return

        weight = self.weight_fn(pl_module).detach()

        if self.bias_fn is not None:
            bias = self.bias_fn(pl_module).detach()
            if bias.dim() == 1:
                bias = bias.unsqueeze(1)
            weight = torch.cat([weight, bias], dim=1)

        plt.figure(figsize=(10, 8))
        sns.heatmap(weight.cpu().numpy(), cmap="coolwarm", center=0)
        plt.xlabel("Output Dimension")
        plt.ylabel("Input Dimension")
        plt.title(f"Weight/Bias Heatmap - Epoch {epoch}")

        image_path = self.save_dir / f"epoch_{epoch:04d}.png"
        plt.savefig(image_path)
        plt.close()

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        image_files = sorted(list(self.save_dir.glob("epoch_*.png")))
        images = [path.as_posix() for path in image_files]
        clip = ImageSequenceClip(images, fps=self.fps)
        video_path = self.save_dir / "weight_evolution.mp4"
        clip.write_videofile(video_path.as_posix(), codec="libx264", audio=False)
        if isinstance(trainer.logger, WandbLogger):
            trainer.logger.log_video(
                key=f"{self.name}_evolution", videos=[video_path.as_posix()]
            )


def embedding_weight_heatmap(
    module: nn.Embedding,
    *,
    save_dir: str | Path,
    name: str,
    fps: int = 30,
    frame_every_n_epochs: int = 1,
) -> WeightBiasHeatmap:
    return WeightBiasHeatmap(
        lambda _: module.weight,
        save_dir=save_dir,
        name=name,
        fps=fps,
        frame_every_n_epochs=frame_every_n_epochs,
    )


def linear_weight_bias_heatmap(
    module: nn.Linear,
    *,
    save_dir: str | Path,
    name: str,
    fps: int = 30,
    frame_every_n_epochs: int = 1,
) -> WeightBiasHeatmap:
    return WeightBiasHeatmap(
        lambda _: module.weight,
        lambda _: module.bias,
        save_dir=save_dir,
        name=name,
        fps=fps,
        frame_every_n_epochs=frame_every_n_epochs,
    )


def rnn_weight_bias_heatmaps(
    module: nn.RNN | nn.LSTM | nn.GRU,
    *,
    save_dir: str | Path,
    name: str,
    fps: int = 30,
    frame_every_n_epochs: int = 1,
) -> List[WeightBiasHeatmap]:
    callbacks = [
        *[
            WeightBiasHeatmap(
                lambda _: getattr(module, f"weight_ih_l{i}"),
                lambda _: getattr(module, f"bias_ih_l{i}"),
                save_dir=save_dir,
                name=f"{name}/weight_ih_l{i}",
                fps=fps,
                frame_every_n_epochs=frame_every_n_epochs,
            )
            for i in range(module.num_layers)
        ],
        *[
            WeightBiasHeatmap(
                lambda _: getattr(module, f"weight_hh_l{i}"),
                lambda _: getattr(module, f"bias_hh_l{i}"),
                save_dir=save_dir,
                name=f"{name}/weight_hh_l{i}",
                fps=fps,
                frame_every_n_epochs=frame_every_n_epochs,
            )
            for i in range(module.num_layers)
        ],
    ]

    if isinstance(module, nn.LSTM):
        callbacks.extend(
            [
                *[
                    WeightBiasHeatmap(
                        lambda _: getattr(module, f"weight_ih_l{i}_reverse"),
                        lambda _: getattr(module, f"bias_ih_l{i}_reverse"),
                        save_dir=save_dir,
                        name=f"{name}/weight_ih_l{i}_reverse",
                        fps=fps,
                        frame_every_n_epochs=frame_every_n_epochs,
                    )
                    for i in range(module.num_layers)
                ],
                *[
                    WeightBiasHeatmap(
                        lambda _: getattr(module, f"weight_hh_l{i}_reverse"),
                        lambda _: getattr(module, f"bias_hh_l{i}_reverse"),
                        save_dir=save_dir,
                        name=f"{name}/weight_hh_l{i}_reverse",
                        fps=fps,
                        frame_every_n_epochs=frame_every_n_epochs,
                    )
                    for i in range(module.num_layers)
                ],
            ]
        )

    return callbacks
