from demos.translation.tokenizer import tokenize_en, tokenize_zh
from demos.translation.vocab import bs, build_vocabulary


def test_token_index():
    vocab_src, vocab_tgt = build_vocabulary(
        lambda: (
            [
                ("今天天气不错。", "The weather is nice today."),
                ("你好！", "Hello!"),
                ("我是一个学生。", "I am a student."),
            ]
        ),
        min_freq=1,
    )
    assert vocab_src.get_itos()[vocab_src[bs]] == bs


def test_try_tokenizer():
    print(tokenize_zh("我是一个学生。"))
    print(tokenize_en("What's your name?"))
    print(tokenize_en("You're the only one."))
