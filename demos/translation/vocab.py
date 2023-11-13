import os
from typing import Tuple

import torch
from spacy.language import Language
from torchtext.vocab import Vocab, build_vocab_from_iterator
from tqdm import tqdm

from demos.translation.data_source import load_data
from demos.translation.tokenizer import tokenize_en, tokenize_zh
from utils import plot_length_info

bs = "<s>"
eos = "</s>"
padding = "<blank>"
unk = "<unk>"
specials = [bs, eos, padding, unk]


def build_vocabulary(load_data, min_freq=2) -> Tuple[Vocab, Vocab]:
    length_info = [{}, {}]

    def yield_tokens(data_iter, tokenizer: Language, index: int):
        for from_to_tuple in tqdm(data_iter):
            tokens = tokenizer(from_to_tuple[index])
            token_length = len(tokens)
            if token_length in length_info[index]:
                length_info[index][token_length] += 1
            else:
                length_info[index][token_length] = 1
            yield tokens

    print("Building Chinese Vocabulary...")
    vocab_src = build_vocab_from_iterator(
        yield_tokens(load_data(), tokenize_zh, index=0),
        min_freq=min_freq,
        specials=specials,
    )

    print("Building English Vocabulary...")
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(load_data(), tokenize_en, index=1),
        min_freq=min_freq,
        specials=specials,
    )
    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    plot_length_info(
        length_info[0],
        "Chinese Sentence Token Length Distribution",
        "output/chinese_length.png",
    )
    plot_length_info(
        length_info[1],
        "English Sentence Token Length Distribution",
        "output/english_length.png",
    )
    return vocab_src, vocab_tgt


def load_vocab(
    vocab_path: str,
    total_size: int,
    min_freq: int = 2,
) -> Tuple[Vocab, Vocab]:
    if not os.path.exists(vocab_path):
        vocab_src, vocab_tgt = build_vocabulary(
            lambda: load_data(0, total_size), min_freq
        )
        torch.save((vocab_src, vocab_tgt), vocab_path)
    else:
        print("%s already exists, load from it." % vocab_path)
        vocab_src, vocab_tgt = torch.load(vocab_path)

    print(f"Vocabulary zh size: {len(vocab_src)}, en size: {len(vocab_tgt)}")
    return vocab_src, vocab_tgt
