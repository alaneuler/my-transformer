import torch
from demos.translation.data_loader import create_data_loader
from demos.translation.data_source import load_test_data
from demos.translation.tokenizer import vocab_tgt
from inference.greedy import greedy_decode
from train.data import Batch

def predict(model):
    pad_idx = vocab_tgt['<blank>']
    test_iter = [
        ('你是谁', 'Of course, the fall of the house of Lehman Brothers has nothing to do with the fall of the Berlin Wall.')
    ]
    data_loader = create_data_loader(test_iter, torch.device('cpu'), 1)

    for test_item in data_loader:
        b = Batch(test_item[0], test_item[1], pad_idx)
        output = greedy_decode(model, b.src, b.src_mask, vocab_tgt['<s>'], 72)[0]
        text = ' '.join(
            [vocab_tgt.get_itos()[x] for x in output if x != pad_idx]
        ).split('</s>', 1)[0]
        print("Predict:", text)
        # print("Groud-Truth:", test_item[1])
