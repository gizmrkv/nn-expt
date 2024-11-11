from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class TupleAutoencoder(L.LightningModule):
    def __init__(
        self,
        tuple_size: int,
        range_size: int,
        *,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        reinforce_loss: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tuple_size = tuple_size
        self.range_size = range_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.reinforce_loss = reinforce_loss

    def log_metrics(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        target: torch.Tensor,
        *,
        prog_bar: bool = False,
        prefix: str = "train/",
    ):
        self.log("loss", loss.mean())

        distr = Categorical(logits=logits)
        action = logits.argmax(dim=-1)
        acc_mean = (action == target).float().mean()
        entropy = distr.entropy().mean()
        self.log(prefix + "acc_mean", acc_mean, prog_bar=prog_bar)
        self.log(prefix + "entropy", entropy)

        zero_pad = len(str(self.tuple_size - 1))
        for i in range(self.tuple_size):
            acc_i = (action[:, i] == target[:, i]).float().mean()
            ent_i = distr.entropy()[:, i].mean()
            self.log(prefix + f"acc_{i:0{zero_pad}}", acc_i)
            self.log(prefix + f"ent_{i:0{zero_pad}}", ent_i)

    def calc_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.reinforce_loss:
            distr = Categorical(logits=logits)
            action = distr.sample()
            log_prob = distr.log_prob(action).sum(dim=-1)
            reward = (action == target).float().mean(dim=-1)
            loss = -(reward - reward.mean()) / (reward.std() + 1e-8) * log_prob
        else:
            loss = (
                F.cross_entropy(
                    logits.view(-1, self.range_size),
                    target.view(-1),
                    reduction="none",
                )
                .view(-1, self.tuple_size)
                .mean(-1)
            )

        return loss.mean()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        input, target = batch
        logits, _ = self(input)
        loss = self.calc_loss(logits, target)
        self.log_metrics(loss, logits, target)
        return loss.mean()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        input, target = batch
        logits, _ = self(input)
        loss = self.calc_loss(logits, target)
        self.log_metrics(loss, logits, target, prog_bar=True, prefix="valid/")
        return loss.mean()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
