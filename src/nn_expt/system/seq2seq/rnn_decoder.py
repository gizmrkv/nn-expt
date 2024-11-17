from typing import List, Literal, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .base import Seq2SeqSystem


class RNNDecoderSeq2SeqSystem(Seq2SeqSystem):
    def __init__(
        self,
        max_length: int,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        rnn_type: Literal["RNN", "LSTM", "GRU"],
        num_layers: int = 1,
        peeky: bool = False,
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
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.peeky = peeky

        self.embedding = nn.Embedding(max_length, embedding_dim)
        self.hidden_linears = nn.ModuleList(
            nn.Linear(max_length * embedding_dim, hidden_size)
            for _ in range(num_layers)
        )
        self.start_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        self.output_linear = nn.Linear(hidden_size, vocab_size)

        rnn_dict = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
        }
        self.rnn = rnn_dict[rnn_type](
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(x).view(-1, self.max_length * self.embedding_dim)
        h_0 = torch.stack([linear(x) for linear in self.hidden_linears])
        h = h_0
        if isinstance(self.rnn, nn.LSTM):
            c = torch.zeros_like(h)

        i = self.start_embedding.repeat(x.size(0), 1)

        symbol_list: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        for _ in range(self.max_length):
            i = i.unsqueeze(1)
            if isinstance(self.rnn, nn.LSTM):
                y, (h, c) = self.rnn(i, (h, c))  # type: ignore
            else:
                y, h = self.rnn(i, h)

            if self.peeky:
                h = h + h_0

            logits = self.output_linear(y.squeeze(1))
            distr = Categorical(logits=logits)

            if self.training:
                x = distr.sample()
            else:
                x = logits.argmax(dim=-1)

            i = self.embedding(x)
            symbol_list.append(x)
            logits_list.append(logits)

        symbols = torch.stack(symbol_list, dim=1)
        logits = torch.stack(logits_list, dim=1)
        return logits, symbols