import copy

import torch
from torch import nn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def model_parameter_size(model: nn.Module):
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    return total_params, trainable_params


class GeneratorWithLength:
    "A generator contains length information, mainly for tqdm."

    def __init__(self, iter, length):
        self.iter = iter
        self.length = length

    def __iter__(self):
        return self.iter

    def __len__(self):
        return self.length
