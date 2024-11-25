import sys
sys.path.extend(['./', '../', '../../'])
import torch
import torch.nn as nn
from torch import Tensor
from traj.GPE.gridE import GridEbd, GxyEbd

class SINW(nn.Module):

    def __init__(self, dim, device, **kw):
        super(SINW, self).__init__()
        self.w1 = nn.Parameter(torch.rand(dim // 4) - 0.5, requires_grad=True).to(device)
        self.w2 = nn.Parameter(torch.rand(dim // 4) - 0.5, requires_grad=True).to(device)
        self.w3 = nn.Parameter(torch.rand(dim // 4) - 0.5, requires_grad=True).to(device)
        self.w4 = nn.Parameter(torch.rand(dim // 4) - 0.5, requires_grad=True).to(device)
        self.dim = dim

    def forward(self, T: Tensor):
        bs = 0 if len(T.shape) == 2 else len(T)
        if bs:
            T = T.reshape(-1, len(T[0][0]))
        x, y = (T[:, 0], T[:, 1])
        xs = x.unsqueeze(1).expand([len(x), 32])
        ys = y.unsqueeze(1).expand([len(y), 32])
        sc = torch.concat([torch.sin(xs * self.w1), torch.cos(xs * self.w2), torch.sin(ys * self.w3), torch.cos(ys * self.w4)], dim=-1)
        if bs:
            sc = sc.reshape(bs, -1, self.dim)
        return sc

class XY(nn.Module):

    def __init__(self, dim, **kw):
        super(XY, self).__init__()
        self.nn = nn.Linear(2, dim)
        self.dim = dim

    def forward(self, T: Tensor):
        bs = 0 if len(T.shape) == 2 else len(T)
        if bs:
            T = T.reshape(-1, len(T[0][0]))
        res = self.nn(T[:, :2])
        if bs:
            res = res.reshape(bs, -1, self.dim)
        return res

class XYVV(nn.Module):

    def __init__(self, dim, device, **kw):
        super(XYVV, self).__init__()
        self.nn = nn.Linear(4, dim)
        self.device = device

    def fwd1(self, T: Tensor):
        """(n*4)=xyvv(n*2)"""
        res = torch.concat([T[:, :2], torch.zeros(len(T), 2).to(self.device)], dim=1)
        res[1:, 2:] = (T[1:, :2] - T[:-1, :2]) / ((T[1:, 2] - T[:-1, 2]).abs() + 0.0001).unsqueeze(-1).expand(len(T) - 1, 2)
        res[0, 2:] = res[1, 2:]
        return self.nn(res)

    def forward(self, T: Tensor):
        """n*2 or bs*n*2"""
        res = self.fwd1(T) if len(T.shape) == 2 else torch.stack([self.fwd1(t) for t in T]).to(self.device)
        return res

class XYDD(nn.Module):

    def __init__(self, dim, device, **kw):
        super(XYDD, self).__init__()
        self.nn = nn.Linear(4, dim)
        self.dim = dim
        self.device = device

    def fwd1(self, T: Tensor):
        """(n*4)=xyvv(n*2)"""
        res = torch.concat([T[:, :2], torch.zeros(len(T), 2).to(self.device)], dim=1)
        res[1:, 2:] = T[1:, :2] - T[:-1, :2]
        res[0, 2:] = res[1, 2:]
        return self.nn(res)

    def forward(self, T: Tensor):
        """n*2 or bs*n*2"""
        return self.fwd1(T) if len(T.shape) == 2 else torch.stack([self.fwd1(t) for t in T]).to(self.device)

class XYRLG(nn.Module):

    def __init__(self, dim, device, city, nxy=None, dxy=None, **kw):
        super(XYRLG, self).__init__()
        self.g = GridEbd(dim // 2, city, nxy, dxy)
        assert self.g._trained
        self.g.requires_grad_(False) 
        self.n1 = nn.Linear(4, dim // 2)
        self.dim = dim
        self.device = device

    def xyrl(self, T: Tensor):
        """n*2 -> n*4"""
        d = torch.norm(T[:-1, :2] - T[1:, :2], dim=1)
        l = torch.zeros(len(T))
        l[0] = d[0]
        l[-1] = d[-1]
        l[1:-1] = (d[1:] + d[:-1]) / 2
        dx, dy = (T[:-1, 0] - T[1:, 0], T[:-1, 1] - T[1:, 1])
        a = torch.arctan2(dy, dx)
        a[torch.where(torch.isnan(a))] = 0
        a[torch.where(torch.isinf(a))] = 0
        da = torch.fmod(a[:-1] + torch.pi * 2 - a[1:], torch.pi * 2)
        res = torch.concat([T[:, :2], torch.zeros(len(T), 2).to(self.device)], dim=1)
        res[:, 2] = l
        res[1:-1, 3] = da
        return res

    def fwd1(self, T: Tensor):
        x = self.xyrl(T)
        n = self.n1(x)
        g = self.g(T, False)
        res = torch.concat([n, g], dim=-1)
        return res

    def forward(self, T: Tensor):
        """n*3 or bs*n*3"""
        res = self.fwd1(T) if len(T.shape) == 2 else torch.stack([self.fwd1(t) for t in T]).to(self.device)
        return res

class XYG(nn.Module):

    def __init__(self, dim, device, city, nxy=None, dxy=None, **kw):
        super(XYG, self).__init__()
        self.g = GridEbd(dim // 2, city, nxy, dxy)
        assert self.g._trained
        self.g.requires_grad_(False) 
        self.n1 = nn.Linear(2, dim // 2)
        self.dim = dim
        self.device = device

    def forward(self, T: Tensor):
        bs = 0 if len(T.shape) == 2 else len(T)
        if bs:
            T = T.reshape(-1, len(T[0][0]))
        res = torch.concat([self.n1(T[:, :2]), self.g(T, False)], dim=-1).to(self.device)
        if bs:
            res = res.reshape(bs, -1, self.dim)
        return res

class XYs(nn.Module):

    def __init__(self, dim, device, **kw):
        super(XYs, self).__init__()
        self.nn = nn.Linear(12, dim)
        self.dim = dim
        self.device = device

    def vv(self, T: Tensor):
        res = torch.zeros((len(T), 2)).to(self.device)
        res[1:] = (T[1:, :2] - T[:-1, :2]) / ((T[1:, 2] - T[:-1, 2]).abs() + 0.0001).unsqueeze(-1).expand(len(T) - 1, 2)
        res[0] = res[1]
        return res

    def dd(self, T: Tensor):
        res = torch.zeros((len(T), 2)).to(self.device)
        res[1:] = T[1:, :2] - T[:-1, :2]
        res[0] = res[1]
        return res

    def dsaj(self, T: Tensor):
        res = torch.zeros((len(T), 4)).to(self.device)
        d = torch.norm(T[:-1, :2] - T[1:, :2], dim=1)
        t = T[:-1, 2] - T[1:, 2] + 1e-09
        s = d / t
        res[1:, 0], res[1:, 1] = (d, s)
        a = (s[1:] - s[:-1]) / t[1:]
        res[2:, 2], res[:2, 2] = (a, a[0])
        j = (a[1:] - a[:-1]) / t[2:]
        res[3:, 3], res[:3, 3] = (j, j[0])
        return res

    def rl(self, T: Tensor):
        """n*2 -> n*4"""
        d = torch.norm(T[:-1, :2] - T[1:, :2], dim=1)
        l = torch.zeros(len(T))
        l[0] = d[0]
        l[-1] = d[-1]
        l[1:-1] = (d[1:] + d[:-1]) / 2
        dx, dy = (T[:-1, 0] - T[1:, 0], T[:-1, 1] - T[1:, 1])
        a = torch.arctan2(dy, dx)
        a[torch.where(torch.isnan(a))] = 0
        a[torch.where(torch.isinf(a))] = 0
        da = torch.fmod(a[:-1] + torch.pi * 2 - a[1:], torch.pi * 2)
        res = torch.zeros((len(T), 2)).to(self.device)
        res[:, 0] = l
        res[1:-1, 1] = da
        return res

    def fwd1(self, T: Tensor):
        x = torch.concat([T[:, :2], self.vv(T), self.dd(T), self.dsaj(T), self.rl(T)], dim=-1).to(self.device)
        x = self.nn(x)
        return x

    def forward(self, T: Tensor):
        """n*3 or bs*n*3"""
        res = self.fwd1(T) if len(T.shape) == 2 else torch.stack([self.fwd1(t) for t in T]).to(self.device)
        return res

class XYSARD(XYs):

    def __init__(self, dim, device, city, nxy=None, dxy=None, **kw):
        super(XYs, self).__init__()
        self.n1 = nn.Linear(6, dim)
        self.dim = dim
        self.device = device

    def fwd1(self, T: Tensor):
        x = torch.concat([T[:, :2], self.dsaj(T)[:, :3], self.rl(T)[:, 0].unsqueeze(-1)], dim=-1)
        return self.n1(x)

    def forward(self, T: Tensor):
        """n*3 or bs*n*3"""
        res = self.fwd1(T) if len(T.shape) == 2 else torch.stack([self.fwd1(t) for t in T]).to(self.device)
        return res