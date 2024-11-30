from typing import List, Literal, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Seq2SeqRNNDecoder(nn.Module):
    def __init__(
        self,
        in_vocab_size: int,
        in_max_length: int,
        out_vocab_size: int,
        out_max_length: int,
        *,
        embedding_dim: int,
        one_hot: bool = False,
        hidden_size: int,
        rnn_type: Literal["rnn", "lstm", "gru"] = "gru",
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.in_vocab_size = in_vocab_size
        self.in_max_length = in_max_length
        self.out_vocab_size = out_vocab_size
        self.out_max_length = out_max_length
        self.one_hot = one_hot
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        if one_hot:
            self.embedding_dim = in_vocab_size
            self.embedding = nn.Embedding(in_vocab_size, in_vocab_size)
            self.embedding.weight.data = torch.eye(in_vocab_size)
            self.embedding.weight.requires_grad = False
        else:
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(in_vocab_size, embedding_dim)
            nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

        self.linear = nn.Linear(
            in_max_length * self.embedding_dim, hidden_size * (1 + bidirectional)
        )
        rnn_types = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        self.rnn = rnn_types[rnn_type](
            self.embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.sos_embedding = nn.Parameter(torch.zeros(1, self.embedding_dim))
        self.output_linear = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.embedding(input)
        emb = emb.view(-1, self.in_max_length * self.embedding_dim)
        h = self.linear(emb)
        h = h.view(self.num_layers * (1 + self.bidirectional), -1, self.hidden_size)

        if self.rnn_type == "lstm":
            h = (h, torch.zeros_like(h))

        x = self.sos_embedding.repeat(input.size(0), 1)
        symbol_list: List[torch.Tensor] = []
        logits_list: List[torch.Tensor] = []
        for _ in range(self.out_max_length):
            x = x.unsqueeze(1)
            y, h = self.rnn(x, h)

            logits = self.output_linear(y.squeeze(1))

            if self.training:
                distr = Categorical(logits=logits)
                x = distr.sample()
            else:
                x = logits.argmax(dim=-1)

            symbol_list.append(x)
            logits_list.append(logits)
            x = self.embedding(x)

        symbols = torch.stack(symbol_list, dim=1)
        logits = torch.stack(logits_list, dim=1)
        return logits, symbols
