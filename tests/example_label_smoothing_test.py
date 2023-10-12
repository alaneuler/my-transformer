import altair as alt
import pandas as pd
import torch
from train.label_smoothing import LabelSmoothing

def example_label_smoothing():
    criterion = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor([
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0],
    ])
    criterion(x=predict.log(), target=torch.LongTensor([
        2, 1, 0, 3, 3
    ]))
    print(criterion.true_dist)
    data = pd.concat([
        pd.DataFrame({
            "Target Distribution": criterion.true_dist[x, y].flatten(),
            "Column": y,
            "Row": x
        })
        for y in range(5)
        for x in range(5)
    ])
    chart = alt.Chart(data).mark_rect().encode(
        alt.X("Column:O", title=None),
        alt.Y("Row:O", title=None),
        alt.Color("Target Distribution:Q", scale=alt.Scale(scheme="viridis"))
    )
    chart.save("output/example_label_smoothing.png", ppi=320)

example_label_smoothing()
