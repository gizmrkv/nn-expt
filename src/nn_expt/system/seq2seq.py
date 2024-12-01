from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Seq2SeqSystem(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        reinforce_loss: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.optimizer = optimizer
        self.reinforce_loss = reinforce_loss

    def step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        *,
        prog_bar: bool = False,
        prefix: str = "train/",
    ) -> torch.Tensor:
        input, target = batch
        logits = self.model(input)

        if isinstance(logits, tuple):
            logits, sequence = logits
        elif self.training:
            distr = Categorical(logits=logits)
            sequence = distr.sample()
        else:
            sequence = logits.argmax(dim=-1)

        loss = self.calc_loss(logits, sequence, target)
        self.log_metrics(
            loss,
            logits,
            sequence,
            target,
            prog_bar=prog_bar,
            prefix=prefix,
        )
        return loss.mean()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, batch_idx, prog_bar=True, prefix="train/")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, batch_idx, prog_bar=True, prefix="val/")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return self.optimizer or torch.optim.Adam(self.model.parameters())

    def log_metrics(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        target: torch.Tensor,
        *,
        prog_bar: bool = False,
        prefix: str = "train/",
    ):
        self.log(prefix + "loss", loss.mean(), prog_bar=prog_bar)

        acc_mean = (sequence == target).float().mean()
        entropy = Categorical(logits=logits).entropy()
        self.log(prefix + "acc_mean", acc_mean)
        self.log(prefix + "entropy", entropy.mean())

        max_length = target.size(-1)
        zero_pad = len(str(max_length - 1))
        for i in range(max_length):
            acc_i = (sequence[:, i] == target[:, i]).float().mean()
            ent_i = entropy[:, i].mean()
            self.log(prefix + f"acc_{i:0{zero_pad}}", acc_i)
            self.log(prefix + f"ent_{i:0{zero_pad}}", ent_i)

        grad_norms = [
            p.grad.norm() for p in self.model.parameters() if p.grad is not None
        ]
        if grad_norms:
            grad_norm = torch.stack(grad_norms).mean().item()
            self.log(prefix + "grad_norm", grad_norm)

    def calc_loss(
        self, logits: torch.Tensor, sequence: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        vocab_size = logits.size(-1)
        max_length = target.size(-1)
        if self.reinforce_loss:
            distr = Categorical(logits=logits)
            log_prob = distr.log_prob(sequence).sum(dim=-1)
            reward = (sequence == target).float().mean(dim=-1)
            loss = -(reward - reward.mean()) / (reward.std() + 1e-8) * log_prob
        else:
            loss = (
                F.cross_entropy(
                    logits.view(-1, vocab_size),
                    target.view(-1),
                    reduction="none",
                )
                .view(-1, max_length)
                .mean(-1)
            )

        return loss
