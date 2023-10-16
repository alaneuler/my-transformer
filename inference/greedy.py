import torch

from utils import subsequent_mask


def greedy_decode(model, src, src_mask, start_symbol, max_len):
    memory = model.encode(src, src_mask)
    y = torch.zeros(1, 1).fill_(start_symbol).type_as(src)
    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, y, subsequent_mask(y.size(1)).type_as(src))
        # Get the last time step result which is the prediction for the next word
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        y = torch.cat([y, torch.zeros(1, 1).type_as(src).fill_(next_word[0])], dim=1)
    return y
