from typing import Any, Dict

import torch
import torch.nn as nn
from torchrl.modules import MLP

from .base import Seq2SeqSystem


class MLPSeq2SeqSystem(Seq2SeqSystem):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        *,
        reinforce_loss: bool = False,
        embedding_dim: int | None = None,
        embedding_init_std: float = 1e-4,
        mlp_kwargs: Dict[str, Any],
        optimizer_class: str,
        optimizer_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.reinforce_loss = reinforce_loss
        self.embedding_init_std = embedding_init_std
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        if embedding_dim is None:
            embedding_dim = vocab_size
            self.embedding = nn.Embedding(vocab_size, vocab_size)
            self.embedding.weight.data = torch.eye(vocab_size)
            self.embedding.weight.requires_grad = False
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=embedding_init_std)

        self.embedding_dim = embedding_dim

        if "activation_class" in mlp_kwargs:
            activation_class = mlp_kwargs.pop("activation_class")
            match activation_class:
                case "identity":
                    mlp_kwargs["activation_class"] = nn.Identity
                case "relu":
                    mlp_kwargs["activation_class"] = nn.ReLU
                case "leaky_relu":
                    mlp_kwargs["activation_class"] = nn.LeakyReLU
                case "tanh":
                    mlp_kwargs["activation_class"] = nn.Tanh
                case "sigmoid":
                    mlp_kwargs["activation_class"] = nn.Sigmoid
                case _:
                    raise ValueError(f"Invalid activation class: {activation_class}")

        if "norm_class" in mlp_kwargs:
            norm_class = mlp_kwargs.pop("norm_class")
            match norm_class:
                case "batch":
                    mlp_kwargs["norm_class"] = nn.BatchNorm1d
                case "layer":
                    mlp_kwargs["norm_class"] = nn.LayerNorm
                case _:
                    raise ValueError(f"Invalid norm class: {norm_class}")

        self.mlp = MLP(
            in_features=self.embedding_dim * seq_len,
            out_features=vocab_size * seq_len,
            **mlp_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.embedding(x)
        logits = self.mlp(emb.view(-1, self.seq_len * self.embedding_dim))
        return logits.view(-1, self.seq_len, self.vocab_size)

    def configure_optimizers(self):
        if self.optimizer_class == "sgd":
            return torch.optim.SGD(self.parameters(), **self.optimizer_kwargs)
        elif self.optimizer_class == "adam":
            return torch.optim.Adam(self.parameters(), **self.optimizer_kwargs)
        else:
            raise ValueError(f"Invalid optimizer class: {self.optimizer_class}")
