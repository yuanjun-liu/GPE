import torch
import torch.nn as nn
import torch.nn.init as init
import math
import random
import numpy as np

def reset_parameters(mod: nn.Module, h) -> None:
    stdv = 1.0 / math.sqrt(h) if h > 0 else 0
    for weight in mod.parameters():
        init.uniform_(weight, -stdv, stdv)

def reset_param_uniform(weight, h):
    stdv = 1.0 / math.sqrt(h) if h > 0 else 0
    init.uniform_(weight, -stdv, stdv)

def freeze(x: nn.Parameter):
    """with optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)"""
    x.requires_grad = False

def dis_mtx(x: torch.Tensor):
    return torch.norm(x[:, None] - x, dim=2, p=2)
_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def auto_device(x: torch.Tensor=None):
    return _device if x is None else x.to(_device)
min_max_var = lambda w: [w.min(), w.max(), torch.var_mean(w)]

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)