import datetime
from pathlib import Path
from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from callback import PlotEmbeddingsCallback, PlotLinearCallback
from data_module import TupleDataModule
from lightning.pytorch.loggers import WandbLogger


class TupleVAE(L.LightningModule):
    def __init__(
        self,
        tuple_length: int,
        range_size: int,
        embedding_dim: int,
        latent_dim: int,
        beta: float = 1.0,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tuple_length = tuple_length
        self.range_size = range_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.lr = lr
        self.weight_decay = weight_decay

        self.embedding = nn.Embedding(range_size, embedding_dim)

        combined_dim = tuple_length * embedding_dim
        self.encoder_mu = nn.Linear(combined_dim, latent_dim)
        self.encoder_logvar = nn.Linear(combined_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, tuple_length * range_size),
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        embedded = self.embedding(x)
        embedded_flat = embedded.view(batch_size, -1)

        mu = self.encoder_mu(embedded_flat)
        logvar = self.encoder_logvar(embedded_flat)

        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        output = self.decoder(z)
        output = output.view(-1, self.tuple_length, self.range_size)
        output = F.softmax(output, dim=-1)
        return output

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        output = self.decode(z)
        return output, mu, logvar

    def _compute_loss(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch[0]
        output, mu, logvar = self(x)

        recon_loss = F.nll_loss(
            output.log().view(-1, self.range_size), x.view(-1), reduction="mean"
        )
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = (recon_loss + self.beta * kl_loss) / x.size(0)

        return total_loss, recon_loss / x.size(0), kl_loss / x.size(0)

    def _compute_metrics(self, x: torch.Tensor, output: torch.Tensor):
        entropy = -(output * torch.log(output + 1e-10)).sum(dim=-1)
        mean_entropy_per_position = entropy.mean(dim=0)
        mean_entropy = entropy.mean()

        sampled = torch.multinomial(output.view(-1, self.range_size), 1).view(x.shape)
        accuracy_per_position = (sampled == x).float().mean(dim=0)
        total_accuracy = (sampled == x).float().mean()

        return {
            "mean_entropy": mean_entropy,
            "total_accuracy": total_accuracy,
            "entropy_per_position": mean_entropy_per_position,
            "accuracy_per_position": accuracy_per_position,
        }

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x = batch[0]
        total_loss, recon_loss, kl_loss = self._compute_loss(batch)
        output, _, _ = self(x)

        metrics = self._compute_metrics(x, output)

        self.log("train_loss", total_loss)
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl_loss", kl_loss)
        self.log("train_mean_entropy", metrics["mean_entropy"])
        self.log("train_accuracy", metrics["total_accuracy"])

        for i, (ent, acc) in enumerate(
            zip(metrics["entropy_per_position"], metrics["accuracy_per_position"])
        ):
            self.log(f"train_entropy_pos_{i}", ent)
            self.log(f"train_accuracy_pos_{i}", acc)

        return total_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        x = batch[0]
        total_loss, recon_loss, kl_loss = self._compute_loss(batch)
        output, _, _ = self(x)

        metrics = self._compute_metrics(x, output)

        self.log("val_loss", total_loss)
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl_loss", kl_loss)
        self.log("val_mean_entropy", metrics["mean_entropy"])
        self.log("val_accuracy", metrics["total_accuracy"])

        for i, (ent, acc) in enumerate(
            zip(metrics["entropy_per_position"], metrics["accuracy_per_position"])
        ):
            self.log(f"val_entropy_pos_{i}", ent)
            self.log(f"val_accuracy_pos_{i}", acc)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


TUPLE_SIZE = 3
RANGE_SIZE = 50
EMBEDDING_DIM = 32
BATCH_SIZE = 2048
NUM_EPOCHS = 2
LATENT_DIM = 32
BETA = 0.01

datamodule = TupleDataModule(
    tuple_size=TUPLE_SIZE, range_size=RANGE_SIZE, batch_size=BATCH_SIZE
)

model = TupleVAE(
    tuple_length=TUPLE_SIZE,
    range_size=RANGE_SIZE,
    embedding_dim=EMBEDDING_DIM,
    latent_dim=LATENT_DIM,
    beta=BETA,
)

run_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
wandb_logger = WandbLogger(run_name, project="tuple-vae")

log_dir = Path("logs") / run_name
plot_embeddings_callback = PlotEmbeddingsCallback(log_dir)
plot_encoder_mu_weights_callback = PlotLinearCallback(
    model.encoder_mu, "encoder_mu", log_dir / "encoder_mu"
)
plot_encoder_logvar_weights_callback = PlotLinearCallback(
    model.encoder_logvar, "encoder_logvar", log_dir / "encoder_logvar"
)

trainer = L.Trainer(
    max_epochs=NUM_EPOCHS,
    accelerator="auto",
    devices=1,
    logger=wandb_logger,
    callbacks=[
        plot_embeddings_callback,
        plot_encoder_mu_weights_callback,
        plot_encoder_logvar_weights_callback,
    ],
)
trainer.fit(model, datamodule)
wandb_logger.finalize("success")
