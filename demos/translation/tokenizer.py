import os
import spacy
import torch
from demos.translation.data_source import load_test_data, load_train_val_data
from itertools import chain
from torchtext.vocab import build_vocab_from_iterator

bs = '<s>'
eos = '</s>'
padding = '<blank>'
unk = '<unk>'
specials = [bs, eos, padding, unk]

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

spacy_zh, spacy_en = load_tokenizers()

def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]

def tokenize_zh(text):
    return tokenize(text, spacy_zh)
def tokenize_en(text):
    return tokenize(text, spacy_en)

def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])

def build_vocabulary(load_data, min_freq=2):
    print("Building Chinese Vocabulary...")
    train, val, test = load_data()
    vocab_src = build_vocab_from_iterator(
        yield_tokens(chain(train, val, test), tokenize_zh, index=0),
        min_freq=min_freq,
        specials=specials
    )

    print("Building English Vocabulary...")
    train, val, test = load_data()
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(chain(train, val, test), tokenize_en, index=1),
        min_freq=min_freq,
        specials=specials
    )
    vocab_src.set_default_index(vocab_src['<unk>'])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt

vocab_path = 'data/vocab.pt'
def load_vocab(min_freq=2):
    if not os.path.exists(vocab_path):
        def load_data():
            train, val = load_train_val_data()
            test = load_test_data()
            return train, val, test

        vocab_src, vocab_tgt = build_vocabulary(load_data, min_freq)
        torch.save((vocab_src, vocab_tgt), vocab_path)
    else:
        print('%s already exists, load from it.' % vocab_path)
        vocab_src, vocab_tgt = torch.load(vocab_path)

    print("Vocabulary zh size:", len(vocab_src), 'en size:', len(vocab_tgt))
    return vocab_src, vocab_tgt
