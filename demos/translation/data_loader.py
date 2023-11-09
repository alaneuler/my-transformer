import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import Vocab

from demos.translation.data_source import load_train_val_data
from demos.translation.tokenizer import tokenize_en, tokenize_zh
from demos.translation.vocab import bs, eos, padding, unk


def collate_batch(
    batch,
    device,
    vocab_src: Vocab,
    vocab_tgt: Vocab,
    max_padding,
    should_check_tokens,
):
    bs_idx = torch.tensor([vocab_src[bs]], device=device)
    eos_idx = torch.tensor([vocab_src[eos]], device=device)
    padding_idx = vocab_src[padding]
    unk_idx = vocab_src[unk]

    def check_tokens(tokens, tokens_idx):
        if not should_check_tokens:
            return

        if len(tokens) > max_padding:
            raise RuntimeError(
                f"Tokens is too long, length {len(tokens)}\ntokens: {tokens}"
            )
        if unk_idx in tokens_idx:
            raise RuntimeError(
                f"unk token in tokens: {tokens}\n, tokens_idx: {tokens_idx}"
            )

    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        src_tokens = tokenize_zh(_src)
        src_tokens_idx = vocab_src(src_tokens)
        check_tokens(src_tokens, src_tokens_idx)
        processed_src = torch.cat(
            [
                bs_idx,
                torch.tensor(
                    src_tokens_idx,
                    dtype=torch.int64,
                    device=device,
                ),
                eos_idx,
            ]
        )
        tgt_tokens = tokenize_en(_tgt)
        tgt_tokens_idx = vocab_tgt(tgt_tokens)
        check_tokens(tgt_tokens, tgt_tokens_idx)
        processed_tgt = torch.cat(
            [
                bs_idx,
                torch.tensor(
                    tgt_tokens_idx,
                    dtype=torch.int64,
                    device=device,
                ),
                eos_idx,
            ]
        )
        src_list.append(
            pad(
                processed_src,
                (0, max_padding - len(processed_src)),
                value=padding_idx,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=padding_idx,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_data_loader(
    data_iter,
    device,
    vocab_src,
    vocab_tgt,
    batch_size,
    max_padding,
    is_distributed,
    should_check_tokens,
):
    def collate_fn(batch):
        return collate_batch(
            batch,
            device,
            vocab_src,
            vocab_tgt,
            max_padding,
            should_check_tokens,
        )

    # Use map style dataset to support distributed training
    data_iter_map = to_map_style_dataset(data_iter)
    data_sampler = DistributedSampler(data_iter_map) if is_distributed else None
    data_loader = DataLoader(
        data_iter_map,
        batch_size=batch_size,
        shuffle=(data_sampler is None),
        sampler=data_sampler,
        collate_fn=collate_fn,
    )
    return data_loader


def create_data_loaders(
    device,
    vocab_src,
    vocab_tgt,
    train_size: int,
    val_size: int,
    batch_size=32,
    max_padding=128,
    is_distributed=False,
    should_check_tokens=True,
):
    train_iter, val_iter = load_train_val_data(train_size, val_size)
    train_data_loader = create_data_loader(
        train_iter,
        device,
        vocab_src,
        vocab_tgt,
        batch_size,
        max_padding,
        is_distributed,
        should_check_tokens,
    )
    val_data_loader = create_data_loader(
        val_iter,
        device,
        vocab_src,
        vocab_tgt,
        batch_size,
        max_padding,
        is_distributed,
        should_check_tokens,
    )
    return train_data_loader, val_data_loader
