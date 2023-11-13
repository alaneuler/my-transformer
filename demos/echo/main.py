import logging

import torch
from torch.optim.lr_scheduler import LambdaLR

from demos.echo.data import data_gen
from inference.greedy import greedy_decode
from model import make_model
from train.label_smoothing import LabelSmoothing
from train.learning_rate import rate
from train.loss import SimpleLossCompute
from train.routine import run_epoch
from utils import model_parameter_size

logger = logging.getLogger("echoLogger")

V = 11
criterion = LabelSmoothing(size=V)
model = make_model(V, V, N=2)
a, t = model_parameter_size(model)
logger.info(f"Model total parameters: {a}")
logger.info(f"Model trainable parameters: {t}")
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
)
lr_scheduler = LambdaLR(
    optimizer=optimizer,
    lr_lambda=lambda step: rate(
        step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
    ),
)

batch_size = 80
for epoch in range(10):
    print("Epoch:", epoch)
    model.train()
    run_epoch(
        model,
        data_gen(V, batch_size, 20),
        SimpleLossCompute(model.generator, criterion),
        optimizer,
        lr_scheduler,
        mode="train",
    )
    model.eval()
    total_loss, total_token = run_epoch(
        model,
        data_gen(V, batch_size, 5),
        SimpleLossCompute(model.generator, criterion),
        None,
        None,
        mode="eval",
    )
    logger.info("Validation result: total Loss: %.2f" % total_loss)
    logger.info(f"Total token: {total_token}")

model.eval()
src = torch.LongTensor([[0, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
max_len = src.shape[1]
src_mask = torch.ones(1, 1, max_len)
print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))
