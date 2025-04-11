import sys, os, torch,time,math
sys.path.extend(['./', '../', '../../'])
from _nn.nLoss import ContrastiveLossInfoNCE
from _nn.nData import auto_device
from _nn.nFile import save_weight, load_weight
from _tool.mFile import is_win, log_dir, out_dir
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import numpy as np
from torch.nn import TransformerEncoder,TransformerEncoderLayer
from traj.GPE.GPE import GPE
from traj.GPE.baseline import *
from traj.GPE.loaddata import migrate_space, dxy, ts_max_len, ts_min_len, traj_mean_len, load_traj
import traj.GPE.loaddata as lld
from _tool.SysMonitor import GroupLogS,LogToTxt
print = GroupLogS(log_dir(), cls=LogToTxt) 
device = auto_device()
num_layer = 2
bidir = True
epoch_max = 20
epoch_min = 2
earlystop = 100
log_int = 10
batch_size = 1024
augment_drop_rate = 0.1
augment_noise_size = 0.0002

class BaseTrajDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, ts, train=True, do_noise=False):
        super().__init__()
        self.ts = ts
        if len(self.ts) < 5:
            self.ts = self.ts[0]
        self.h = min(traj_mean_len[dataset], ts_max_len)
        self.dim = len(self.ts[0][0])
        self.len_mask = [torch.Tensor(np.array([1] * i + [0] * (ts_max_len - i))) for i in range(ts_max_len)]
        self.train = train
        self.do_noise = do_noise

    def __len__(self):
        return len(self.ts)

    def _augment(self, t: torch.Tensor):
        le = len(t)
        drop_num = int(max(0, le - ts_min_len) * augment_drop_rate)
        idx = torch.Tensor(np.random.choice(le, drop_num, replace=False)).long()
        mask = torch.ones(le).bool()
        mask[idx] = False
        t = t[mask]
        shift_xy = torch.rand(len(t), 2) * augment_noise_size
        t[:, :2] += shift_xy
        return t

    def __getitem__(self, i):
        """return ts,len,mask  + ts_pos,len_pos,mask_pos if self.train"""
        t = self.ts[i]
        le = len(t)
        t = torch.Tensor(t)
        tr = torch.zeros((ts_max_len, self.dim))
        tr[:le] = t
        if not self.train:
            if self.do_noise:
                t2 = self._augment(t)
                le2 = len(t2)
                tr2 = torch.zeros((ts_max_len, self.dim))
                tr2[:le2] = t2
                return (tr2, le2, self.len_mask[le2 - 1])
            return (tr, le, self.len_mask[le - 1])
        t2 = self._augment(t)
        le2 = len(t2)
        tr2 = torch.zeros((ts_max_len, self.dim))
        tr2[:le2] = t2
        return (tr, le, self.len_mask[le - 1], tr2, le2, self.len_mask[le2 - 1])


def cut_len(ts, ls):
    return ts[:, :max(ls)]

def _lin_net(nlayer=3, dim_in=128, dim_out=128):
    dim = max(dim_in, dim_out)
    if nlayer == 2: return nn.Sequential(nn.Linear(dim_in, dim), nn.SiLU(), nn.Linear(dim, dim_out))
    elif nlayer == 3: return nn.Sequential(nn.Linear(dim_in, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim_out))
    return nn.Identity()

def _num_nn(name):
    d = {'XY': 2, 'XYVV': 2, 'XYDD': 2, 'XYRLG': 2, 'XYG': 2,  'XYSARD': 2}
    return d[name] if name in d else 3

class BaseTfModel(nn.Module):
    def __init__(self, ebd: nn.Module, ename, dim_ebd, dim_hid, dim_out, num_layer=2, bidir=True, device=auto_device()) -> None:
        super(BaseTfModel, self).__init__()
        self.ebd =nn.Sequential(ebd, _lin_net(_num_nn(ename), dim_ebd, dim_ebd))
        self.dim_hid, self.bidir, self.num_layer, self.device = (dim_hid, bidir, num_layer, device)
        
        den = torch.exp(torch.arange(0, dim_ebd, 2) * (-math.log(10000)) / dim_ebd)
        pos = torch.arange(0, ts_max_len).reshape(ts_max_len, 1)
        pos_embedding = torch.zeros((ts_max_len, dim_ebd))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0) # 1(bs) len dim
        self.pos_ebd=pos_embedding.to(device)

        encoder_layer = TransformerEncoderLayer(
                d_model=dim_ebd,
                nhead=1,
                dim_feedforward=dim_hid*2,
                dropout=0,
                activation=torch.nn.functional.relu,
                layer_norm_eps=1e-5,
                batch_first=False,
                norm_first=False,
                bias=True,
                device=device,
            )
        encoder_norm = torch.nn.LayerNorm(dim_hid, eps=1e-5, bias=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layer, encoder_norm)

        self.dim2 = dim_hid * 2 if bidir else dim_hid
        self.l1 = nn.Sequential(nn.BatchNorm1d(self.dim2), nn.Linear(self.dim2, self.dim2), nn.ReLU())
        self.l2 = nn.Linear(self.dim2, dim_out)
        
    def forward(self, x: torch.Tensor, len_mask: torch.Tensor):
        x = self.ebd(x) # bs len dim
        len_mask = len_mask.unsqueeze(-1).expand(x.shape) # bs len dim
        x+=self.pos_ebd[:,:x.size(1),:]
        x = x * len_mask
        x = x.transpose(0, 1).contiguous() # len bs dim
        memory = self.encoder(x) # transformer encoder
        y=torch.concat([memory[0],memory[-1]],dim=-1) if self.bidir else memory[0]
        y = self.l2(self.l1(y)) # bs*dim
        return y

def name2ebd(method_name: str, city: str,dim) -> nn.Module:
    kw = {'device': device, 'dim': dim, 'city': city, 'nxy': None, 'dxy': dxy(city)}
    if 'GPE' in method_name:
        return GPE(dim=dim, name=method_name, device=device)
    elif 'SINW' in method_name:
        return SINW(**kw)
    elif 'XYVV' in method_name:
        return XYVV(**kw)
    elif 'XYDD' in method_name:
        return XYDD(**kw)
    elif 'XYRLG' in method_name:
        return XYRLG(**kw)
    elif 'XYSARD' in method_name:
        return XYSARD(**kw)
    elif 'XYG' in method_name:
        return  XYG(**kw)
    elif 'XY' in method_name:
        return  XY(**kw)
    elif 'Grid' in method_name:
        return GridEbd(**kw)
    elif 'Gxy' in method_name:
        return GxyEbd(**kw)

def _tf_model_path(method_name: str, train_city: str,dim):
    ebd: nn.Module = name2ebd(method_name, train_city,dim)
    model = BaseTfModel(ebd, method_name, dim, dim, dim, num_layer=num_layer, bidir=bidir)
    path = os.path.join(out_dir('ckpt_tf'), f'{method_name}_{train_city}_{lld.nTrain}-{lld.nTest}_d{dim}n{num_layer}b{(1 if bidir else 0)}_d{augment_drop_rate}n{augment_noise_size}_l{ts_max_len}{ts_min_len}b{batch_size}e{epoch_min}s{earlystop}.th')
    if os.path.exists(path): load_weight(model, path)
    return (model, path)

def train_tf(method_name: str, train_city: str,dim):
    model, path = _tf_model_path(method_name, train_city,dim)
    if os.path.exists(path): return True
    print.set(method_name + str(dim)+'/' + train_city + '-' + str(lld.nTrain), newfile=True)
    print('train_method', method_name, dim, train_city, lld.nTrain)
    model = model.to(device)
    model.train()
    TS = load_traj(train_city)[0]
    dataset = BaseTrajDataset(train_city, TS)
    dataloder = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=0 if is_win else 4)
    lossf = ContrastiveLossInfoNCE(device=device, batch_size=batch_size).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()))
    bi, best_bi, best_loss, best_w = (0, 0, 10000000, None)
    for ep in range(epoch_max):
        for ts, les, masks, ts_p, les_p, masks_p in dataloder:
            ts, ts_p = (cut_len(ts, les), cut_len(ts_p, les_p))
            masks, masks_p = (cut_len(masks, les), cut_len(masks_p, les_p))
            ts, masks, ts_p, masks_p = (ts.to(device), masks.to(device), ts_p.to(device), masks_p.to(device))
            vs, vs_pos = (model(ts, masks), model(ts_p, masks_p))
            loss: Tensor = lossf(vs, vs_pos)
            losv = loss.item()
            if bi % log_int == 0:
                print(method_name+f' ep:{ep}, batch:{bi}, loss:{losv}, ')
                if losv < best_loss:
                    best_bi, best_loss = (bi, losv)
                    save_weight(model, path)
            if ep >= epoch_min and bi - best_bi > earlystop:
                model.cpu()
                print('finish train_method', method_name, train_city)
                return False
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10)  
            opt.step()
            bi += 1
    model.cpu()
    print('finish train_method', method_name, train_city)
    print.unset()
    return False

def infer_tf(method_name, train_city, to_city, TS, dim,do_noise=False) -> Tensor:
    """load(model,train_city), adapt(to_city), return model(ts)"""
    model, path = _tf_model_path(method_name, train_city,dim)
    assert os.path.exists(path), 'Model untrained'
    model.eval()
    model.to(device)
    if 'GPE' not in method_name:
        TS = migrate_space(train_city, to_city, TS)
    res = []
    with torch.no_grad():
        for ts, les, masks in DataLoader(BaseTrajDataset(to_city, TS, train=False, do_noise=do_noise), batch_size=batch_size, shuffle=False, drop_last=False):
            ts, masks = (cut_len(ts, les), cut_len(masks, les))
            ts, masks = (ts.to(device), masks.to(device))
            vs = model(ts, masks).detach().cpu()
            res.append(vs)
    model.cpu()
    res = torch.concat(res, dim=0)
    return res