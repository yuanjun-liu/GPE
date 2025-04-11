import sys, os
sys.path.extend(['./', '../', '../../'])
from _nn.nData import auto_device
import torch
PI = torch.pi
import torch.nn as nn
import numpy as np
from torch import Tensor

def _parse_name(name:str):
    ss=name.split('_')
    e1=float(ss[1])
    return e1

# name: GPE_(\epslion/2pi), note the e slightly differs from the paper
class GPE(nn.Module):
    def __init__(self, dim=128, name='GPE_1e-06', device=auto_device()) -> None:
        """dim>=4, e/2pi \in[0,1) """
        super(GPE, self).__init__()
        assert dim % 4 == 0
        e=_parse_name(name)
        ' multi base '
        jz_len = [1]
        jz_base = [1]
        self.ws = [torch.ones(1).to(device) / 180 * PI]
        jz_power = np.zeros(dim)
        jz_power[0] = 1
        p = 1
        while True:
            while jz_power[p - 1]:
                p += 1
            # for j in range(int(np.log(dim) / np.log(p))):
            #     jz_power[p ** j - 1] = 1
            for j in range(dim):
                jzp=p**j-1
                if jzp>=dim:break
                jz_power[jzp]=1
            le = int(np.ceil(-np.log(e) / np.log(p)))
            le = max(le, 1)
            le = min(le, dim // 4 - sum(jz_len))
            self.ws.append(Tensor([p ** (i + 1) for i in range(le)]).to(device) / 180 * PI)
            jz_len.append(le)
            jz_base.append(p)
            if sum(jz_len) >= dim // 4:
                break
        self.dim = dim

    def forward(self, T: Tensor):
        """T.shape=(n,2) or (bs,n,2)"""
        bs = 0 if len(T.shape) == 2 else len(T)
        if bs:
            T = T.reshape(-1, len(T[0][0]))
        lat, lon = (T[:, 0], T[:, 1])
        wlats, wlons = ([], [])
        for w in self.ws:
            wlats.append(lat.unsqueeze(1).expand([len(lat), len(w)]) * w)
            wlons.append(lon.unsqueeze(1).expand([len(lon), len(w)]) * w)
        wlons, wlats = (torch.concat(wlons, dim=-1), torch.concat(wlats, dim=-1))
        sc = torch.concat([torch.sin(wlons), torch.cos(wlons), torch.sin(wlats), torch.cos(wlats)], dim=-1)
        if bs:
            sc = sc.reshape(bs, -1, self.dim)
        return sc
