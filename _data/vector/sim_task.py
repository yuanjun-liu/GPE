import sys
sys.path.extend(['./', '../../', '../'])
import torch
from _nn.nMath import sim_mtx, dis_mtx

def knni(sim: torch.Tensor, k: int):
    """sim[q][d]: |Q|*|D| ; knn(q) in D ; return |Q|*k"""
    return torch.topk(sim, k, dim=-1).indices

def knnvd(VQ: torch.Tensor, VD: torch.Tensor, k: int):
    """Q: n*dim ; knn(q) in D ; return |Q|*k"""
    return knni(-dis_mtx(VQ, VD), k)

def knn_union(k1: torch.Tensor, k2: torch.Tensor):
    """k1,k2:n*k ; return 01(n*k) of k1 ; precise=sum(return)/(n*k)"""
    n1, nk1 = k1.shape
    n2, nk2 = k2.shape
    assert n1 == n2
    b1 = torch.unsqueeze(k1, -1).expand(n1, nk1, nk2)
    b2 = torch.unsqueeze(k2, -2).expand(n1, nk1, nk2)
    return torch.sum(b1 == b2, dim=-1)

def knn_prec(k1: torch.Tensor, k2: torch.Tensor):
    """k1,k2:n*k ; return float"""
    n, k = k1.shape
    ku = knn_union(k1, k2)
    return ku.sum() / (n * k)

def ranki(dis):
    """TS->A,B ; rank(a) in B ; dis[a][b] ; return |Q|"""
    idx_sort = torch.sort(dis).indices
    a2b = torch.arange(0, len(dis)).unsqueeze(-1).expand(dis.shape).to(idx_sort.device)
    return torch.where(idx_sort == a2b)[1] + 1

def rankvs(VA: torch.Tensor, VB: torch.Tensor):
    """TS->A,B: n*dim ; rank(a) in B ; return |Q|"""
    return ranki(-sim_mtx(VA, VB))