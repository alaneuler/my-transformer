import os
from typing import List, Tuple

import spacy
import torch
from spacy.language import Language
from torchtext.vocab import Vocab, build_vocab_from_iterator
from tqdm import tqdm

from demos.translation.data_source import load_data

bs = "<s>"
eos = "</s>"
padding = "<blank>"
unk = "<unk>"
specials = [bs, eos, padding, unk]


def load_tokenizers() -> Tuple[Language, Language]:
    try:
        spacy_zh = spacy.load("zh_core_web_sm")
    except IOError:
        os.system("python -m spacy download zh_core_web_sm")
        spacy_zh = spacy.load("zh_core_web_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_zh, spacy_en


spacy_zh, spacy_en = load_tokenizers()


def tokenize_zh(text: str) -> List[str]:
    return [tok.text for tok in spacy_zh.tokenizer(text)]


def tokenize_en(text: str) -> List[str]:
    return [tok.text for tok in spacy_en.tokenizer(text)]


def build_vocabulary(load_data, min_freq=2) -> Tuple[Vocab, Vocab]:
    def yield_tokens(data_iter, tokenizer: Language, index: int):
        for from_to_tuple in tqdm(data_iter):
            yield tokenizer(from_to_tuple[index])

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

    print("Vocabulary zh size:", len(vocab_src), "en size:", len(vocab_tgt))
    return vocab_src, vocab_tgt
