from typing import Dict, Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class Seq2SeqSystem(L.LightningModule):
    reinforce_loss: bool

    def step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        *,
        log_prefix: str = "train/",
    ) -> Dict[str, torch.Tensor]:
        input, target = batch
        logits = self(input)

        if isinstance(logits, tuple):
            logits, sequence = logits
        elif self.training:
            distr = Categorical(logits=logits)
            sequence = distr.sample()
        else:
            sequence = logits.argmax(dim=-1)

        loss = self.calc_loss(logits, sequence, target)
        metrics = self.log_metrics(
            loss=loss,
            logits=logits,
            sequence=sequence,
            target=target,
            prefix=log_prefix,
        )
        metrics["loss"] = loss.mean()
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
        *,
        loss: torch.Tensor,
        logits: torch.Tensor,
        sequence: torch.Tensor,
        target: torch.Tensor,
        prefix: str = "train/",
    ) -> Dict[str, torch.Tensor]:
        metrics: Dict[str, torch.Tensor] = {}
        metrics[prefix + "loss"] = loss.mean()

        acc_mean = (sequence == target).float().mean()
        entropy = Categorical(logits=logits).entropy()
        metrics[prefix + "acc_mean"] = acc_mean
        metrics[prefix + "entropy"] = entropy.mean()

        seq_len = target.size(-1)
        zero_pad = len(str(seq_len - 1))
        for i in range(seq_len):
            acc_i = (sequence[:, i] == target[:, i]).float().mean()
            ent_i = entropy[:, i].mean()
            metrics[prefix + f"acc_{i:0{zero_pad}}"] = acc_i
            metrics[prefix + f"ent_{i:0{zero_pad}}"] = ent_i

        for k, v in metrics.items():
            self.log(k, v.mean(), prog_bar=k.endswith("loss"))

        return metrics

    def on_after_backward(self):
        norms = {}
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                norms[f"grad/l2_norm_{name}"] = param.grad.norm(p=2)

        for k, v in norms.items():
            self.log(k, v)

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
