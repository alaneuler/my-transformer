import altair as alt
import pandas as pd
import torch

from positional_encoding import PositionalEncoding


def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [0, 2, 4, 6]
        ]
    )
    chart = (
        alt.Chart(data)
        .encode(x="position", y="embedding", color="dimension:N")
        .mark_line()
    )
    chart.save("./output/example_positional.png", ppi=320)


example_positional()
