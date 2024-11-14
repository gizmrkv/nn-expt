from typing import Tuple

import torch
import torch.nn as nn

from .tuple2tuple import Tuple2TupleSystem


class Tuple2TupleLinearSystem(Tuple2TupleSystem):
    def __init__(
        self,
        tuple_size: int,
        range_size: int,
        embedding_dim: int,
        *,
        lr: float = 0.001,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            tuple_size=tuple_size,
            range_size=range_size,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(range_size, embedding_dim)
        self.linear = nn.Linear(tuple_size * embedding_dim, tuple_size * range_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        emb: torch.Tensor = self.embedding(x)
        emb = emb.view(-1, self.tuple_size * self.embedding_dim)
        logits: torch.Tensor = self.linear(emb)
        logits = logits.view(-1, self.tuple_size, self.range_size)
        return logits, None
