import pandas as pd
import torch
from positional_encoding import PositionalEncoding

def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe(torch.zeros(1, 100, 20))
    print(y.shape)

    data = pd.concat([
        pd.DataFrame({
            'embedding': y[0, :, dim],
            'dimension': dim,
            'position': list(range(100))
        })
        for dim in [4, 5, 6, 7]
    ])
    print(data)


example_positional()
