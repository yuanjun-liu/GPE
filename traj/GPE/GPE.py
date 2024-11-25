import sys, os
sys.path.extend(['./', '../', '../../'])
from _tool.mFile import out_dir, log_dir
from _nn.nLoss import ContrastiveLossInfoNCE
from _nn.nData import auto_device
from _nn.nFile import load_weight
import torch
PI = torch.pi
import torch.nn as nn
import numpy as np
from torch import Tensor
from _tool.SysMonitor import GroupLogS
_print = print
print = GroupLogS(log_dir())

def _lin_net(nlayer=3, dim_in=128, dim_out=128):
    dim = max(dim_in, dim_out)
    if nlayer == 2: return nn.Sequential(nn.Linear(dim_in, dim), nn.SiLU(), nn.Linear(dim, dim_out))
    elif nlayer == 3: return nn.Sequential(nn.Linear(dim_in, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim_out))
    return nn.Identity()


def _parse_name(name:str):
    """input: GPE_e1 or GPEwT_e1_e2, output e1,e2"""
    ss=name.split('_')
    e1=float(ss[1])
    e2=float(ss[2]) if len(ss)>2 else None
    return e1,e2

# name: GPE_(\epslion/2pi)_\xi, note the e slightly differs from the paper
class GPE(nn.Module):
    def __init__(self, dim=128, name='GPE_1e-06_0.0001', device=auto_device()) -> None:
        """dim>=4, e/2pi \in[0,1) """
        super(GPE, self).__init__()
        assert dim % 4 == 0
        e=_parse_name(name)[0]
        assert e <= 1, ValueError('e_gpe must <=1, so -log(e)>=0')
        self.path = os.path.join(out_dir('GPE'), f'{name}_{dim}.th')
        self.nn = _lin_net(3 if 'wT' in name else 0, dim, dim)
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
            for j in range(int(np.log(dim) / np.log(p))):
                jz_power[p ** j - 1] = 1
            le = int(np.ceil(-np.log(e) / np.log(p)))
            le = max(le, 1)
            le = min(le, dim // 4 - sum(jz_len))
            self.ws.append(Tensor([p ** (i + 1) for i in range(le)]).to(device) / 180 * PI)
            jz_len.append(le)
            jz_base.append(p)
            if sum(jz_len) >= dim // 4:
                break
        self.dim = dim
        if os.path.exists(self.path):
            load_weight(self, self.path)
            self.requires_grad_(False)

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
        return self.nn(sc)

class _GPE_Trainer:
    def __init__(self, name, dim=128, bs=1024, device=auto_device(), log_ep=1000):
        self.bs, self.device, self.name = (bs, device, name)
        self.name=name;self.log_ep = log_ep
        self.lossf = ContrastiveLossInfoNCE(bs, device=device).to(device)
        self.gpe: GPE = GPE(dim-dim,name=name,device=device)
        epd = {'1k': 1000, '1b': 100,'': 10000,}
        self.earlystop = [epd[i] for i in epd if i in name][0]
        self.trained = os.path.exists(self.gpe.path)
        self.noise_range = _parse_name(name)[1]
        self.opt = torch.optim.AdamW(self.gpe.parameters())
        self.best_loss = 9999

    def gendata(self):
        ps = torch.rand([self.bs, 2]) * 360
        ps -= 180
        ps[:, 0] /= 2
        ps2 = ps + torch.rand_like(ps) * self.noise_range
        return [ps, ps2]

    def train_one(self):
        ps, ps2 = self.gendata()
        ps, ps2 = (ps.to(self.device), ps2.to(self.device))
        v1 = self.gpe(ps)
        v2 = self.gpe(ps2)
        loss = self.lossf(v1, v2)
        losv = loss.item()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return losv

    def train(self):
        self.gpe.train()
        bi, best_bi = (0, 0)
        train_lossv = 0.5
        while True:
            train_lossv = self.train_one()
            if train_lossv < self.best_loss:
                best_bi, self.best_loss = (bi, train_lossv)
                torch.save(self.gpe.state_dict(), self.gpe.path)
                print(self.name+ f' best batch:{bi}, train loss:{train_lossv}')
            if bi - best_bi > self.earlystop:
                break
            bi += 1
        self.gpe.cpu()
        torch.save(self.gpe.state_dict(), self.gpe.path)

def gpe_pretrain(name, dim=128, bs=1024, device=auto_device(), log_ep=1000):
    if not 'wT' in name:
        print(name, 'do not need train')
        return
    print.set('p' + name, True)
    trainer = _GPE_Trainer(name, dim, bs, device, log_ep=log_ep)
    if trainer.trained:
        print.unset()
        print(name + ' trained!!')
        return
    print('\n\n\n\ngpe_pretrain', name, dim, bs)
    trainer.train()
    print.unset()
