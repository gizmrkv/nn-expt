from pathlib import Path
from typing import List

import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from moviepy.editor import ImageSequenceClip


class EmbeddingHeatmap(L.Callback):
    def __init__(
        self,
        embedding: nn.Embedding,
        *,
        save_dir: str | Path,
        fps: int = 30,
    ):
        super().__init__()
        self.embedding = embedding
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.image_paths: List[Path] = []

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        epoch = trainer.current_epoch
        weights = self.embedding.weight.detach().cpu().numpy()

        plt.figure(figsize=(10, 8))
        sns.heatmap(weights.T, cmap="viridis")
        plt.xlabel("Embedding Dimension")
        plt.ylabel("Element Index")
        plt.title(f"Embedding Heatmap - Epoch {epoch}")

        image_path = self.save_dir / f"epoch_{epoch:04d}.png"
        plt.savefig(image_path)
        plt.close()

        self.image_paths.append(image_path)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        video_path = self.save_dir / "embedding_evolution.mp4"

        images = [plt.imread(str(path)) for path in sorted(self.image_paths)]
        clip = ImageSequenceClip(images, fps=self.fps)
        clip.write_videofile(str(video_path))
