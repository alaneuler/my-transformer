from typing import Iterator, Tuple

from utils import GeneratorWithLength

zh_file_path = "data/zh-en/news-commentary-v13.zh-en.zh"
en_file_path = "data/zh-en/news-commentary-v13.zh-en.en"


def load_train_val_data(train_size: int, val_size: int):
    return (
        load_data(0, train_size),
        load_data(train_size, train_size + val_size),
    )


def load_test_data(train_size: int, val_size: int, test_size: int):
    return load_data(train_size + val_size, train_size + val_size + test_size)


def load_data(start: int, end: int):
    return GeneratorWithLength(do_load_data(start, end), end - start)


def do_load_data(start: int, end: int) -> Iterator[Tuple[str, str]]:
    "Return type is a list of tuple in format: (Chinese, English)."
    with open(zh_file_path, "r") as zh_file, open(en_file_path, "r") as en_file:
        for i, (zh_line, en_line) in enumerate(zip(zh_file, en_file)):
            if i < start:
                continue
            if i >= end:
                break

            yield zh_line.strip(), en_line.strip()
