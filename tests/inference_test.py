import torch

from model import make_model
from utils import subsequent_mask


def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()

    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    memory = test_model.encode(src, src_mask)
    y = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, y, subsequent_mask(y.size(1)).type_as(src)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        y = torch.cat(
            [y, torch.empty(1, 1).type_as(src).fill_(next_word[0])], dim=1
        )

    print("Example untrained model prediction:", y)


for _ in range(10):
    inference_test()
