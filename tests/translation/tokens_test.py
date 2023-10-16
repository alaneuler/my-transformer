from demos.translation.tokenizer import build_vocabulary


def token_index():
    vocab_src, vocab_tgt = build_vocabulary(
        lambda: (
            [("今天天气不错。", "The weather is nice today."), ("你好！", "Hello!")],
            [],
            [],
        ),
        min_freq=1,
    )
    assert vocab_src.get_itos()[vocab_src["<s>"]] == "<s>"
    print(vocab_src["<s>"])
    print(vocab_src.get_itos())
    print(vocab_tgt.get_itos())
    print(vocab_src.get_stoi())
    print(vocab_tgt.get_stoi())


token_index()
