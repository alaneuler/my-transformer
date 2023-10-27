import torch
from torch import nn
from torchtext.vocab import Vocab

from demos.translation.arguments import ModelArguments
from demos.translation.train_model import translation_model


def load_model(
    model_args: ModelArguments, vocab_src: Vocab, vocab_tgt: Vocab
) -> nn.Module:
    model = translation_model(
        len(vocab_src), len(vocab_tgt), model_args.d_model, model_args.N
    )
    model.load_state_dict(torch.load(model_args.model_name_or_path))
    return model
