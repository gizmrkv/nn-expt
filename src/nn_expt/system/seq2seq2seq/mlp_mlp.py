import itertools
from typing import Any, Dict

import torch
import torch.nn as nn
from torchrl.modules import MLP

from .base import Seq2Seq2SeqSystem


class MLPMLPSeq2Seq2SeqSystem(Seq2Seq2SeqSystem):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        z_vocab_size: int,
        z_seq_len: int,
        *,
        sender_entropy_weight: float = 0.0,
        sender_embedding_dim: int | None = None,
        sender_embedding_init_std: float = 1e-4,
        sender_mlp_kwargs: Dict[str, Any],
        receiver_embedding_dim: int | None = None,
        receiver_embedding_init_std: float = 1e-4,
        receiver_mlp_kwargs: Dict[str, Any],
        optimizer_class: str,
        sender_optimizer_kwargs: Dict[str, Any],
        receiver_optimizer_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.z_vocab_size = z_vocab_size
        self.z_seq_len = z_seq_len
        self.sender_entropy_weight = sender_entropy_weight
        self.sender_embedding_init_std = sender_embedding_init_std
        self.sender_mlp_kwargs = sender_mlp_kwargs
        self.receiver_embedding_init_std = receiver_embedding_init_std
        self.receiver_mlp_kwargs = receiver_mlp_kwargs
        self.optimizer_class = optimizer_class
        self.sender_optimizer_kwargs = sender_optimizer_kwargs
        self.receiver_optimizer_kwargs = receiver_optimizer_kwargs

        if sender_embedding_dim is None:
            sender_embedding_dim = vocab_size
            self.sender_embedding = nn.Embedding(vocab_size, vocab_size)
            self.sender_embedding.weight.data = torch.eye(vocab_size)
            self.sender_embedding.weight.requires_grad = False
        else:
            self.sender_embedding = nn.Embedding(vocab_size, sender_embedding_dim)
            nn.init.normal_(
                self.sender_embedding.weight,
                mean=0.0,
                std=sender_embedding_init_std,
            )

        if receiver_embedding_dim is None:
            receiver_embedding_dim = z_vocab_size
            self.receiver_embedding = nn.Embedding(z_vocab_size, z_vocab_size)
            self.receiver_embedding.weight.data = torch.eye(z_vocab_size)
            self.receiver_embedding.weight.requires_grad = False
        else:
            self.receiver_embedding = nn.Embedding(z_vocab_size, receiver_embedding_dim)
            nn.init.normal_(
                self.receiver_embedding.weight,
                mean=0.0,
                std=receiver_embedding_init_std,
            )

        self.sender_embedding_dim = sender_embedding_dim
        self.receiver_embedding_dim = receiver_embedding_dim

        def prepare_mlp_kwargs(mlp_kwargs: Dict[str, Any]) -> Dict[str, Any]:
            if "activation_class" in mlp_kwargs:
                activation_class = mlp_kwargs.pop("activation_class")
                match activation_class:
                    case "relu":
                        mlp_kwargs["activation_class"] = nn.ReLU
                    case "leaky_relu":
                        mlp_kwargs["activation_class"] = nn.LeakyReLU
                    case "tanh":
                        mlp_kwargs["activation_class"] = nn.Tanh
                    case "sigmoid":
                        mlp_kwargs["activation_class"] = nn.Sigmoid
                    case _:
                        raise ValueError(
                            f"Invalid activation class: {activation_class}"
                        )

            if "norm_class" in mlp_kwargs:
                norm_class = mlp_kwargs.pop("norm_class")
                match norm_class:
                    case "batch":
                        mlp_kwargs["norm_class"] = nn.BatchNorm1d
                    case "layer":
                        mlp_kwargs["norm_class"] = nn.LayerNorm
                    case _:
                        raise ValueError(f"Invalid norm class: {norm_class}")

            return mlp_kwargs

        self.sender_mlp = MLP(
            in_features=sender_embedding_dim * seq_len,
            out_features=z_vocab_size * z_seq_len,
            **prepare_mlp_kwargs(sender_mlp_kwargs),
        )
        self.receiver_mlp = MLP(
            in_features=receiver_embedding_dim * z_seq_len,
            out_features=vocab_size * seq_len,
            **prepare_mlp_kwargs(receiver_mlp_kwargs),
        )

    def sender(self, x: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.sender_embedding(x)
        logits = self.sender_mlp(
            emb.view(-1, self.sender_embedding_dim * self.seq_len),
        )
        return logits.view(-1, self.z_seq_len, self.z_vocab_size)

    def receiver(self, z: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.receiver_embedding(z)
        logits = self.receiver_mlp(
            emb.view(-1, self.receiver_embedding_dim * self.z_seq_len),
        )
        return logits.view(-1, self.seq_len, self.vocab_size)

    def configure_optimizers(self):
        optimizer_params = [
            {
                "params": itertools.chain(
                    self.sender_embedding.parameters(),
                    self.sender_mlp.parameters(),
                ),
                **self.sender_optimizer_kwargs,
            },
            {
                "params": itertools.chain(
                    self.receiver_embedding.parameters(),
                    self.receiver_mlp.parameters(),
                ),
                **self.receiver_optimizer_kwargs,
            },
        ]
        if self.optimizer_class == "sgd":
            return torch.optim.SGD(optimizer_params)
        elif self.optimizer_class == "adam":
            return torch.optim.Adam(optimizer_params)
        else:
            raise ValueError(f"Invalid optimizer class: {self.optimizer_class}")
