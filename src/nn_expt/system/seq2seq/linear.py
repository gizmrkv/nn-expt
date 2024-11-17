from typing import Tuple

import torch
import torch.nn as nn

from .base import Seq2SeqSystem


class LinearSeq2SeqSystem(Seq2SeqSystem):
    def __init__(
        self,
        max_length: int,
        vocab_size: int,
        embedding_dim: int,
        *,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        reinforce_loss: bool = False,
    ):
        super().__init__(
            max_length,
            vocab_size,
            lr=lr,
            weight_decay=weight_decay,
            reinforce_loss=reinforce_loss,
        )
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(max_length * embedding_dim, max_length * vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        emb: torch.Tensor = self.embedding(x)
        emb = emb.view(-1, self.max_length * self.embedding_dim)
        logits: torch.Tensor = self.linear(emb)
        logits = logits.view(-1, self.max_length, self.vocab_size)
        return logits, None
