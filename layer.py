import torch
from torch import nn
from torch.nn.functional import log_softmax

class LayerNorm(nn.Module):
    "Construct a layer norm module."

    def __init__(self, features, eps=1e-6):
        super().__init__()
        # a_2 and b_2 are trainable parameters.
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection.
    Norm is applied first as opposed to the original paper (last).
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class Generator(nn.Module):
    """
    Define a standard linear + log_softmax generation step.
    Generator is connected to the output of the decoder.
    """

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        "Use log_softmax for working in log domain which is numerically more stable"
        return log_softmax(self.proj(x), dim=-1)
