import copy
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
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


def plot_length_info(length_dict: Dict[int, int], title: str, save_path: str):
    df = pd.DataFrame(list(length_dict.items()), columns=["length", "freq"])
    df = df.sort_values(by="length")
    ax = df.plot(kind="bar", x="length", y="freq")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.title(title)

    for index, label in enumerate(ax.xaxis.get_ticklabels()):
        if index % 5 != 0:
            label.set_visible(False)
    plt.xticks(rotation=45)

    plt.savefig(save_path)
    plt.close()
