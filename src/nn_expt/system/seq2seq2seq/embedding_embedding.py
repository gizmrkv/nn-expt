from typing import Any, Dict

import torch
import torch.nn as nn

from .base import Seq2Seq2SeqSystem


class EmbeddingEmbeddingSeq2Seq2SeqSystem(Seq2Seq2SeqSystem):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        *,
        optimizer_class: str,
        sender_optimizer_kwargs: Dict[str, Any],
        receiver_optimizer_kwargs: Dict[str, Any],
        embedding_init_std: float = 1e-4,
        sender_entropy_weight: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.optimizer_class = optimizer_class
        self.sender_optimizer_kwargs = sender_optimizer_kwargs
        self.receiver_optimizer_kwargs = receiver_optimizer_kwargs
        self.embedding_init_std = embedding_init_std
        self.sender_entropy_weight = sender_entropy_weight

        self.sender_embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.normal_(self.sender_embedding.weight, mean=0.0, std=embedding_init_std)

        self.receiver_embedding = nn.Embedding(vocab_size, vocab_size)
        nn.init.normal_(
            self.receiver_embedding.weight, mean=0.0, std=embedding_init_std
        )

    def sender(self, x: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.sender_embedding(x)
        logits = emb.view(-1, self.seq_len, self.vocab_size)
        return logits

    def receiver(self, z: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.receiver_embedding(z)
        logits = emb.view(-1, self.seq_len, self.vocab_size)
        return logits

    def configure_optimizers(self):
        optimizer_params = [
            {
                "params": self.sender_embedding.parameters(),
                **self.sender_optimizer_kwargs,
            },
            {
                "params": self.receiver_embedding.parameters(),
                **self.receiver_optimizer_kwargs,
            },
        ]
        if self.optimizer_class == "sgd":
            return torch.optim.SGD(optimizer_params)
        elif self.optimizer_class == "adam":
            return torch.optim.Adam(optimizer_params)
        else:
            raise ValueError(f"Invalid optimizer class: {self.optimizer_class}")
