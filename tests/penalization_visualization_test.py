import altair as alt
import pandas as pd
import torch
from train.label_smoothing import LabelSmoothing

def loss(x, criterion):
    d = x + 3
    predict = torch.FloatTensor([
        [1e-9, x/d - 1e-9, 1/d, 1/d, 1/d],
    ])
    return criterion(predict.log(), torch.LongTensor([1]))

def penalization_visualization():
    criterion = LabelSmoothing(5, 0, 0.1)
    data = pd.DataFrame({
        "Loss": [
            loss(x, criterion)
            for x in range(1, 100)
        ],
        "Steps": list(range(1, 100))
    }).astype('float')

    chart = alt.Chart(data).mark_line().encode(
        x="Steps",
        y="Loss",
    )
    chart.save("output/penalization_visualization.png", ppi=320)

penalization_visualization()
