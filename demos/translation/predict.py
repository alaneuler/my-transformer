import torch
from demos.translation.data_loader import create_data_loader
from demos.translation.tokenizer import bs, padding
from inference.greedy import greedy_decode
from train.data import Batch

def predict(model, config):
    vocab_src, vocab_tgt = config['vocab_src'], config['vocab_tgt']
    max_padding = config['max_padding']

    pad_idx = vocab_tgt[padding]
    test_iter = [
        ('当然，现在的情况和1989年的情况明显不同了。', ''),
        ('当富人不再那么富了，穷人就会更穷。', '')
    ]
    data_loader = create_data_loader(test_iter, torch.device('cpu'),
                                     vocab_src, vocab_tgt, 1)

    for i, test_item in enumerate(data_loader):
        b = Batch(test_item[0], test_item[1], pad_idx)
        output = greedy_decode(model, b.src, b.src_mask, vocab_tgt[bs], max_padding)
        text = ' '.join(
            [vocab_tgt.get_itos()[x] for x in output[0] if x != pad_idx]
        ).split('</s>', 1)[0]
        print("Translation of:", test_iter[i][0])
        print(text[4:].replace(' ,', ',').replace(' .', '.'))
