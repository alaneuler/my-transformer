import torch
from demos.translation.data_source import load_train_val_data
from demos.translation.tokenizer import tokenize_en, tokenize_zh, vocab_src, vocab_tgt
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtext.data.functional import to_map_style_dataset

def collate_batch(batch,
                  device,
                  max_padding=128,
                  pad_id=2):
    bs_id = torch.tensor([vocab_src['<s>']], device=device)
    eos_id = torch.tensor([vocab_src['</s>']], device=device)

    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        src_tokens = tokenize_zh(_src)
        processed_src = torch.cat([
                bs_id,
                torch.tensor(
                    vocab_src(src_tokens),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id])
        processed_tgt = torch.cat([
                bs_id,
                torch.tensor(
                    vocab_tgt(tokenize_en(_tgt)),
                    dtype=torch.int64,
                    device=device,
                ),
                eos_id])
        src_list.append(
            pad(processed_src,
                (0, max_padding - len(processed_src)),
                value=pad_id))
        tgt_list.append(
            pad(processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id))

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)

def create_data_loader(data_iter, device, batch_size,
                       max_padding=128, is_distributed=False):
    def collate_fn(batch):
        return collate_batch(
            batch,
            device,
            max_padding=max_padding,
            pad_id=vocab_src['<blank>']
        )
    
    data_iter_map = to_map_style_dataset(data_iter)
    data_sampler = DistributedSampler(data_iter_map) if is_distributed else None
    data_loader = DataLoader(
        data_iter_map,
        batch_size=batch_size,
        shuffle=(data_sampler is None),
        sampler=data_sampler,
        collate_fn=collate_fn
    )
    return data_loader

def create_data_loaders(device,
                       batch_size=100,
                       max_padding=128,
                       is_distributed=False):
    train_iter, val_iter = load_train_val_data()
    train_data_loader = create_data_loader(train_iter, device, batch_size,
                                           max_padding, is_distributed)
    val_data_loader = create_data_loader(val_iter, device, batch_size,
                                         max_padding, is_distributed)
    return train_data_loader, val_data_loader
