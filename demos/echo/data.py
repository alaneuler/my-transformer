import torch

from train.data import Batch


def data_gen(V, batch_size, n_batches):
    for i in range(n_batches):
        data = torch.randint(1, V, size=(batch_size, 10))
        # Set the first element of sequence to 0 (start_symbol).
        data[:, 0] = 0
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, pad=0)
