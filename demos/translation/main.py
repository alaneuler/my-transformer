import os

import torch

from demos.translation.predict import predict
from demos.translation.tokenizer import load_vocab
from demos.translation.train_model import train_model, translation_model

model_path = "models/zh_en_final.pt"
d_model = 512
N = 2
min_vocab_freq = 1
vocab_src, vocab_tgt = load_vocab(min_vocab_freq)
config = {
    "d_model": d_model,
    "N": N,
    "device": torch.device("cpu"),
    "batch_size": 2,
    "distributed": False,
    "num_epochs": 30,
    "accum_iter": 1,
    "base_lr": 0.1,
    "max_padding": 72,
    "warmup": 10,
    "model_path": model_path,
    "vocab_src": vocab_src,
    "vocab_tgt": vocab_tgt,
}

if not os.path.exists(model_path):
    train_model(config)

model = translation_model(len(vocab_src), len(vocab_tgt), d_model, N)
model.load_state_dict(torch.load(model_path))

predict(model, config)
