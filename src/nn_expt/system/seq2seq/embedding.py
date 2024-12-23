from typing import Any, Dict

import torch
import torch.nn as nn

from .base import Seq2SeqSystem


class EmbeddingSeq2SeqSystem(Seq2SeqSystem):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        *,
        optimizer_class: str,
        optimizer_kwargs: Dict[str, Any],
        reinforce_loss: bool = False,
        embedding_init_std: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.reinforce_loss = reinforce_loss
        self.embedding_init_std = embedding_init_std

        self.embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=embedding_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.embedding(x)
        logits = emb.view(-1, self.seq_len, self.vocab_size)
        return logits

    def configure_optimizers(self):
        if self.optimizer_class == "sgd":
            return torch.optim.SGD(self.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_class == "adam":
            return torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)
        else:
            raise ValueError(f"Invalid optimizer class: {self.optimizer_class}")
