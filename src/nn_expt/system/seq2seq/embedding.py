from typing import Tuple

import torch
import torch.nn as nn

from .base import Seq2SeqSystem


class EmbeddingSeq2SeqSystem(Seq2SeqSystem):
    def __init__(
        self,
        max_length: int,
        vocab_size: int,
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
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.embedding(x), None
