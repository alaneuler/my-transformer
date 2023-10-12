import torch
from torch import nn

class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        # Saving the true_dist is rather a hack way
        self.true_dist = None
    
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.clone()
        # -2 means all except start_symbol and target
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        # Invalid label if target index is on padding_idx
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            # Fill zero to invalid label
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)
