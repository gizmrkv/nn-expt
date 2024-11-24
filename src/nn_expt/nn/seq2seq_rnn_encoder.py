from typing import Literal

import torch
import torch.nn as nn


class Seq2SeqRNNEncoder(nn.Module):
    def __init__(
        self,
        in_vocab_size: int,
        out_vocab_size: int,
        out_max_length: int,
        *,
        embedding_dim: int,
        hidden_size: int,
        rnn_type: Literal["rnn", "lstm", "gru"] = "gru",
        num_layers: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.out_max_length = out_max_length
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        rnn_types = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}
        self.embedding = nn.Embedding(in_vocab_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0, 1.0)
        self.encoder = rnn_types[rnn_type](
            embedding_dim,
            hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.decoder = nn.Linear(hidden_size, out_vocab_size * out_max_length)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(input)
        zeros = torch.zeros(
            self.num_layers * (1 + self.bidirectional),
            input.size(0),
            self.hidden_size,
            device=input.device,
        )
        if self.rnn_type == "lstm":
            h = (zeros, zeros)
        else:
            h = zeros

        _, h = self.encoder(emb, h)
        if self.rnn_type == "lstm":
            h = h[0]

        if self.bidirectional:
            h = h[-1] + h[-2]
        else:
            h = h[-1]

        return self.decoder(h).view(-1, self.out_max_length, self.out_vocab_size)
