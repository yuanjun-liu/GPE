import sys
sys.path.extend(['./', '../'])
from _nn.nData import auto_device
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np

class ContrastiveLossInfoNCE(nn.Module):

    def __init__(self, batch_size, t=1, device=auto_device()):
        super(ContrastiveLossInfoNCE, self).__init__()
        self.t = t
        self.batch_neg_mask = torch.Tensor(np.array([[1] * i + [0] + [1] * (batch_size - i - 1) for i in range(batch_size)])).bool().to(device)
        self.device = device

    def forward(self, anchor: Tensor, postive: Tensor, negatives=None):
        """anchor:n*d, postive:n*d or n*k*d, negatives:n*d or n*k*d or auto-build"""
        ps = postive.shape
        assert 0 < len(ps) <= 3, 'unsupport dim'
        if negatives is None:
            'anchor:n*d, postive:n*d or n*k*d ; return negatives:n*(2n-2)*d'
            ps = postive.shape
            bs = len(anchor)
            negs = []
            negs.append(torch.stack([anchor[self.batch_neg_mask[i]] for i in range(bs)]))
            if len(ps) == 2:
                negs.append(torch.stack([postive[self.batch_neg_mask[i]] for i in range(bs)]))
            else:
                for k in range(ps[1]):
                    negs.append(torch.stack([postive[:, k, :][self.batch_neg_mask[i]] for i in range(bs)]))
            negatives = torch.concat(negs, dim=1).to(self.device)
        ns = negatives.shape
        assert 0 < len(ns) <= 3, 'unsupport dim'
        assert anchor.shape[-1] == ps[-1] == ns[-1], 'unmatched dim'
        if len(ps) == 2:
            pos = torch.exp(torch.cosine_similarity(anchor, postive) / self.t)
        elif len(ps) == 3:
            pos = torch.exp(torch.cosine_similarity(anchor.unsqueeze(1), postive, dim=-1).sum(dim=-1) / self.t)
        if len(negatives.shape) == 2:
            negs = torch.exp(torch.cosine_similarity(anchor, negatives) / self.t)
        elif len(negatives.shape) == 3:
            negs = torch.exp(torch.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1) / self.t).sum(dim=-1)
        res = -torch.log(pos / (pos + negs)).mean()
        return res