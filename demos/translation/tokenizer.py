import os
import spacy
import torch
from itertools import chain
from load_data import load_data
from torchtext.vocab import build_vocab_from_iterator

def load_tokenizers():
    try:
        spacy_zh = spacy.load('zh_core_web_sm')
    except IOError:
        os.system('python -m spacy download zh_core_web_sm')
        spacy_zh = spacy.load('zh_core_web_sm')
    
    try:
        spacy_en = spacy.load('en_core_web_sm')
    except IOError:
        os.system('python -m spacy download en_core_web_sm')
        spacy_en = spacy.load('en_core_web_sm')

    return spacy_zh, spacy_en

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def build_vocabulary(spacy_zh, spacy_en):
    def tokenize_zh(text):
        return tokenize(text, spacy_zh)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    # train has size 2000, and val has size 100
    print("Building Chinese Vocabulary...")
    train, val = load_data(0, 2000), load_data(2000, 2100)
    vocab_src = build_vocab_from_iterator(
        yield_tokens(chain(train, val), tokenize_zh, index=0),
        min_freq=2,
        specials=['<s>', '</s>', '<blank>', '<unk>']
    )
    print("Building English Vocabulary...")
    train, val = load_data(0, 2000), load_data(2000, 2100)
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(chain(train, val), tokenize_en, index=1),
        min_freq=2,
        specials=['<s>', '</s>', '<blank>', '<unk>']
    )
    vocab_src.set_default_index(vocab_src['<unk>'])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt

vocab_path = 'data/vocab.pt'
def load_vocab(spacy_zh, spacy_en):
    if not os.path.exists(vocab_path):
        vocab_src, vocab_tgt = build_vocabulary(spacy_zh, spacy_en)
        torch.save((vocab_src, vocab_tgt), vocab_path)
    else:
        vocab_src, vocab_tgt = torch.load(vocab_path)

    print("Vocabulary zh size:", len(vocab_src), 'en size:', len(vocab_tgt))
    return vocab_src, vocab_tgt

spacy_de, spacy_en = load_tokenizers()
vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
