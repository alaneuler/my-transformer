import torch
from torchtext.vocab import Vocab

from demos.translation.arguments import ModelArguments
from demos.translation.data_loader import create_data_loader
from demos.translation.vocab import bs, eos, padding
from inference.greedy import greedy_decode
from train.data import Batch


def concatenate_result(model_output, vocab_tgt, bs, eos, pad_idx):
    result = ""
    for x in model_output:
        if x == pad_idx:
            continue
        chr = vocab_tgt.get_itos()[x]
        if chr == eos:
            break
        if chr == bs:
            continue
        if chr == "," or chr == "." or chr == "?":
            result += chr
        else:
            result += " " + chr
    return result[1:]


def predict(
    model, model_args: ModelArguments, vocab_src: Vocab, vocab_tgt: Vocab
):
    pad_idx = vocab_tgt[padding]
    while True:
        query = input("Input a Chinese sentence to translate:\n")
        data = create_data_loader(
            [(query, "")],
            torch.device("cpu"),
            vocab_src,
            vocab_tgt,
            1,
            model_args.max_padding,
            False,
            False,
        )
        for q in data:
            b = Batch(q[0], q[1], pad_idx)
            output = greedy_decode(
                model, b.src, b.src_mask, vocab_tgt[bs], model_args.max_padding
            )
            text = concatenate_result(output[0], vocab_tgt, bs, eos, pad_idx)
            print(text)
