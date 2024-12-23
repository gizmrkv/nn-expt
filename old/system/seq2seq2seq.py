from typing import Dict, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Seq2Seq2SeqSystem(L.LightningModule):
    sender_entropy_weight: bool

    def step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        *,
        log_prefix: str = "train/",
    ) -> Dict[str, torch.Tensor]:
        input, target = batch
        logits_s = self.sender(input)

        if isinstance(logits_s, tuple):
            logits_s, z = logits_s
        elif self.training:
            distr = Categorical(logits=logits_s)
            z = distr.sample()
        else:
            z = logits_s.argmax(dim=-1)

        logits_r = self.receiver(z)

        if isinstance(logits_r, tuple):
            logits_r, sequence = logits_r
        elif self.training:
            distr = Categorical(logits=logits_r)
            sequence = distr.sample()
        else:
            sequence = logits_r.argmax(dim=-1)

        loss_s, loss_r = self.calc_loss(logits_s, logits_r, z, target)
        metrics = self.log_metrics(
            loss_s,
            loss_r,
            logits_s,
            logits_r,
            z,
            sequence,
            target,
            prefix=log_prefix,
        )
        metrics["loss"] = (loss_s + loss_r).mean()
        return metrics

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, log_prefix="train/")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        return self.step(batch, batch_idx, log_prefix="val/")

    def log_metrics(
        self,
        sender_loss: torch.Tensor,
        receiver_loss: torch.Tensor,
        sender_logits: torch.Tensor,
        receiver_logits: torch.Tensor,
        z: torch.Tensor,
        sequence: torch.Tensor,
        target: torch.Tensor,
        *,
        prefix: str = "train/",
    ):
        metrics: Dict[str, torch.Tensor] = {}
        metrics[prefix + "sender_loss"] = sender_loss.mean()
        metrics[prefix + "receiver_loss"] = receiver_loss.mean()
        metrics[prefix + "loss"] = (sender_loss + receiver_loss).mean()

        acc_mean = (sequence == target).float().mean()
        entropy_s = Categorical(logits=sender_logits).entropy()
        entropy_r = Categorical(logits=receiver_logits).entropy()
        metrics[prefix + "acc_mean"] = acc_mean
        metrics[prefix + "sender_entropy"] = entropy_s.mean()
        metrics[prefix + "receiver_entropy"] = entropy_r.mean()

        z_max_length = z.size(-1)
        zero_pad = len(str(z_max_length - 1))
        for i in range(z_max_length):
            ent_i = entropy_s[:, i].mean()
            metrics[prefix + f"sender_ent_{i:0{zero_pad}}"] = ent_i

        max_length = target.size(-1)
        zero_pad = len(str(max_length - 1))
        for i in range(max_length):
            acc_i = (sequence[:, i] == target[:, i]).float().mean()
            ent_i = entropy_r[:, i].mean()
            metrics[prefix + f"acc_{i:0{zero_pad}}"] = acc_i
            metrics[prefix + f"receiver_ent_{i:0{zero_pad}}"] = ent_i

        for k, v in metrics.items():
            self.log(k, v.mean(), prog_bar=k.endswith("receiver_loss"))

        return metrics

    def on_after_backward(self):
        norms = {}
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                norms[f"grad/l2_norm_{name}"] = param.grad.norm(p=2)

        for k, v in norms.items():
            self.log(k, v)

    def calc_loss(
        self,
        sender_logits: torch.Tensor,
        receiver_logits: torch.Tensor,
        z: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vocab_size = receiver_logits.size(-1)
        max_length = target.size(-1)
        loss_r = (
            F.cross_entropy(
                receiver_logits.view(-1, vocab_size),
                target.view(-1),
                reduction="none",
            )
            .view(-1, max_length)
            .mean(-1)
        )
        distr = Categorical(logits=sender_logits)
        log_prob = distr.log_prob(z).sum(dim=-1)
        reward = -loss_r.detach()
        loss_s = -(reward - reward.mean()) / (reward.std() + 1e-8) * log_prob
        entropy_s = distr.entropy().mean()
        loss_s = loss_s - self.sender_entropy_weight * entropy_s
        return loss_s, loss_r
