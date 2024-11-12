from pathlib import Path
from typing import Callable, List

import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from moviepy.editor import ImageSequenceClip


class WeightBiasHeatmap(L.Callback):
    def __init__(
        self,
        weight_fn: Callable[[L.LightningModule], torch.Tensor],
        bias_fn: Callable[[L.LightningModule], torch.Tensor] | None = None,
        *,
        save_dir: str | Path,
        fps: int = 30,
    ):
        super().__init__()
        self.weight_fn = weight_fn
        self.bias_fn = bias_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.image_paths: List[Path] = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        epoch = trainer.current_epoch
        weight = self.weight_fn(pl_module).detach()

        if self.bias_fn is not None:
            bias = self.bias_fn(pl_module).detach()
            if bias.dim() == 1:
                bias = bias.unsqueeze(1)
            weight = torch.cat([weight, bias], dim=1)

        plt.figure(figsize=(10, 8))
        sns.heatmap(weight)
        plt.xlabel("Feature Dimension")
        plt.ylabel("Output Dimension")
        plt.title(f"Weight/Bias Heatmap - Epoch {epoch}")

        image_path = self.save_dir / f"epoch_{epoch:04d}.png"
        plt.savefig(image_path)
        plt.close()

        self.image_paths.append(image_path)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        video_path = self.save_dir / "weight_evolution.mp4"
        images = [plt.imread(str(path)) for path in sorted(self.image_paths)]
        clip = ImageSequenceClip(images, fps=self.fps)
        clip.write_videofile(str(video_path))
