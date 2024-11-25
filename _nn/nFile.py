import torch
import torch.nn as nn

def load_weight(model: nn.Module, path: str):
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage, weights_only=True))

def save_weight(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)