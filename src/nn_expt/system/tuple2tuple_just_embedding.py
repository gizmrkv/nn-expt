from typing import Tuple

import torch
import torch.nn as nn

from .tuple2tuple import Tuple2TupleSystem


class Tuple2TupleJustEmbeddingSystem(Tuple2TupleSystem):
    def __init__(
        self,
        tuple_size: int,
        range_size: int,
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
        self.embedding = nn.Embedding(range_size, range_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.embedding(x), None
