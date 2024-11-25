import torch

def dis_mtx(A: torch.Tensor, B: torch.Tensor):
    """return |A|*|B|"""
    (nq, dim), nd = (A.shape, len(B))
    q = torch.unsqueeze(A, 1).expand(nq, nd, dim)
    d = torch.unsqueeze(B, 0).expand(nq, nd, dim)
    return torch.norm(q - d, dim=-1)

def sim_mtx(A: torch.Tensor, B: torch.Tensor):
    """return |A|*|B|"""
    (nq, dim), nd = (A.shape, len(B))
    q = torch.unsqueeze(A, 1).expand(nq, nd, dim)
    d = torch.unsqueeze(B, 0).expand(nq, nd, dim)
    return torch.cosine_similarity(q, d, dim=-1)