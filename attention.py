import math

import torch
from torch import nn

from utils import clones


def attention(query, key, value, mask=None, dropout=None):
    "The returned tensor is of size (batch_size, 8, seq_len, d_k)."
    d_k = query.size(-1)
    # In the setup of paper, d_k and d_v both equal to 64.
    # query is of size (batch_size, 8, seq_len, 64)
    # key is also of size (batch_size, 8, seq_len, 64)
    # scores are of size (batch_size, 8, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # p_attn is of size (batch_size, 8, seq_len, seq_len)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        n_batches = query.size(0)
        query, key, value = [
            linear(x).view(n_batches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        x = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(n_batches, -1, self.h * self.d_k)
        )
        del query, key, value
        return self.linears[-1](x)
