import os
from typing import List, Tuple

import spacy
from spacy.language import Language


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
