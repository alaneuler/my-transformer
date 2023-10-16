from torch import nn

from layer import LayerNorm, SublayerConnection
from utils import clones


class Encoder(nn.Module):
    "Encoder is a stack of N layers."

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        # The norm operation is not present in the original paper.
        return self.norm(x)


class EncoderLayer(nn.Module):
    "And encoder layer is made up of self-attention and feed-forward network."

    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
