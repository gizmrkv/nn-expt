import torch
import torchrl.modules
from torch import nn


class Seq2SeqLinear(nn.Module):
    def __init__(
        self,
        in_vocab_size: int,
        in_max_length: int,
        out_vocab_size: int,
        out_max_length: int,
        *,
        embedding_dim: int = 32,
        one_hot: bool = False,
        noisy: bool = False,
    ):
        super().__init__()

        self.in_vocab_size = in_vocab_size
        self.in_max_length = in_max_length
        self.out_vocab_size = out_vocab_size
        self.out_max_length = out_max_length
        self.one_hot = one_hot

        if one_hot:
            self.embedding_dim = in_vocab_size
            self.embedding = nn.Embedding(in_vocab_size, in_vocab_size)
            self.embedding.weight.data = torch.eye(in_vocab_size)
            self.embedding.weight.requires_grad = False
        else:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(in_vocab_size, embedding_dim)
            nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

        if noisy:
            self.linear = torchrl.modules.NoisyLinear(
                in_max_length * self.embedding_dim, out_max_length * out_vocab_size
            )
        else:
            self.linear = nn.Linear(
                in_max_length * self.embedding_dim, out_max_length * out_vocab_size
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb: torch.Tensor = self.embedding(x)
        emb = emb.view(-1, self.in_max_length * self.embedding_dim)
        logits: torch.Tensor = self.linear(emb)
        logits = logits.view(-1, self.out_max_length, self.out_vocab_size)
        return logits
