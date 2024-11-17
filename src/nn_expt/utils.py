import datetime

import torch
from torch import nn


def get_run_name() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def init_weights(m: nn.Module):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.RNN, nn.LSTM, nn.GRU)):
        if isinstance(m.weight_ih_l0, torch.Tensor):
            nn.init.xavier_normal_(m.weight_ih_l0)
        if isinstance(m.weight_hh_l0, torch.Tensor):
            nn.init.xavier_normal_(m.weight_hh_l0)
        if isinstance(m.bias_ih_l0, torch.Tensor):
            nn.init.zeros_(m.bias_ih_l0)
        if isinstance(m.bias_hh_l0, torch.Tensor):
            nn.init.zeros_(m.bias_hh_l0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
