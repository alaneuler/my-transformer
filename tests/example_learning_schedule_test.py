import altair as alt
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR

from train.learning_rate import rate


def example_learning_schedule():
    opts = [[512, 1, 4000], [512, 1, 8000], [256, 1, 4000]]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    for example in opts:
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        for _ in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)
    alt.data_transformers.disable_max_rows()

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size & warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )
    chart = (
        alt.Chart(data)
        .mark_line()
        .encode(x="step", y="Learning Rate", color="model_size & warmup:N")
    )
    chart.save("output/learning_schedule.png", ppi=320)


example_learning_schedule()
