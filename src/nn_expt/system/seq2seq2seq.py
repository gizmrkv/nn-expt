from typing import Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Seq2Seq2SeqSystem(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        *,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        encode_entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr
        self.weight_decay = weight_decay
        self.encode_entropy_weight = encode_entropy_weight

    def step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        *,
        prog_bar: bool = False,
        prefix: str = "train/",
    ) -> torch.Tensor:
        input, target = batch
        encode_logits = self.encoder(input)

        if isinstance(encode_logits, tuple):
            encode_logits, z = encode_logits
        elif self.training:
            distr = Categorical(logits=encode_logits)
            z = distr.sample()
        else:
            z = encode_logits.argmax(dim=-1)

        decode_logits = self.decoder(z)

        if isinstance(decode_logits, tuple):
            decode_logits, sequence = decode_logits
        elif self.training:
            distr = Categorical(logits=decode_logits)
            sequence = distr.sample()
        else:
            sequence = decode_logits.argmax(dim=-1)

        encode_loss, decode_loss = self.calc_loss(
            encode_logits, decode_logits, z, target
        )
        self.log_metrics(
            encode_loss,
            decode_loss,
            encode_logits,
            decode_logits,
            z,
            sequence,
            target,
            prog_bar=prog_bar,
            prefix=prefix,
        )
        return (encode_loss + decode_loss).mean()

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, batch_idx, prog_bar=True, prefix="train/")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, batch_idx, prefix="val/")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def log_metrics(
        self,
        encode_loss: torch.Tensor,
        decode_loss: torch.Tensor,
        encode_logits: torch.Tensor,
        decode_logits: torch.Tensor,
        z: torch.Tensor,
        sequence: torch.Tensor,
        target: torch.Tensor,
        *,
        prog_bar: bool = False,
        prefix: str = "train/",
    ):
        self.log(prefix + "encode_loss", encode_loss.mean())
        self.log(prefix + "decode_loss", decode_loss.mean())
        self.log(prefix + "loss", (encode_loss + decode_loss).mean())

        acc_mean = (sequence == target).float().mean()
        encode_entropy = Categorical(logits=encode_logits).entropy()
        decode_entropy = Categorical(logits=decode_logits).entropy()
        self.log(prefix + "acc_mean", acc_mean, prog_bar=prog_bar)
        self.log(prefix + "encode_entropy", encode_entropy.mean())
        self.log(prefix + "decode_entropy", decode_entropy.mean())

        z_max_length = z.size(-1)
        zero_pad = len(str(z_max_length - 1))
        for i in range(z_max_length):
            ent_i = encode_entropy[:, i].mean()
            self.log(prefix + f"encode_ent_{i:0{zero_pad}}", ent_i)

        max_length = target.size(-1)
        zero_pad = len(str(max_length - 1))
        for i in range(max_length):
            acc_i = (sequence[:, i] == target[:, i]).float().mean()
            ent_i = decode_entropy[:, i].mean()
            self.log(prefix + f"acc_{i:0{zero_pad}}", acc_i)
            self.log(prefix + f"decode_ent_{i:0{zero_pad}}", ent_i)

    def calc_loss(
        self,
        encode_logits: torch.Tensor,
        decode_logits: torch.Tensor,
        z: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        vocab_size = decode_logits.size(-1)
        max_length = target.size(-1)
        decode_loss = (
            F.cross_entropy(
                decode_logits.view(-1, vocab_size),
                target.view(-1),
                reduction="none",
            )
            .view(-1, max_length)
            .mean(-1)
        )
        distr = Categorical(logits=encode_logits)
        log_prob = distr.log_prob(z).sum(dim=-1)
        reward = -decode_loss.detach()
        encode_loss = -(reward - reward.mean()) / (reward.std() + 1e-8) * log_prob
        encode_entropy = distr.entropy().mean()
        encode_loss = encode_loss - self.encode_entropy_weight * encode_entropy
        return encode_loss, decode_loss
