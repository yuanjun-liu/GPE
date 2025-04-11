import sys, os, torch
sys.path.extend(['./', '../', '../../'])
from _nn.nData import auto_device
from _nn.nFile import load_weight
from _tool.mFile import out_dir
from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn as nn
import numpy as np
from traj.GPE.model_lstm import cut_len,BaseLSTMModel,name2ebd
from traj.GPE.loaddata import migrate_space, ts_max_len, ts_min_len, traj_mean_len, rank_split_data,datasets
import traj.GPE.loaddata as ldd
device = auto_device()
num_layer = 2
bidir = True
epoch_max = 20
epoch_min = 2
earlystop = 100
log_int = 10
batch_size = 1024
augment_drop_rate = 0.1
augment_noise_size = 0.0003
augment_noise_rate=1

class BaseTrajDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, ts, train=True, do_noise=True):
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
        le = len(t)
        shift_xy = torch.rand(le, 2) * augment_noise_size
        noise_num = int(max(0, le - ts_min_len) * augment_noise_rate)
        idx = torch.Tensor(np.random.choice(le, noise_num, replace=False)).long()
        mask = torch.ones(le).bool()
        mask[idx] = False
        t[mask, :2] += shift_xy[mask]
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

def _trained_model_path(method_name: str, train_city: str,dim):
    ebd: nn.Module = name2ebd(method_name, train_city,dim)
    model = BaseLSTMModel(ebd, method_name, dim, dim, dim, num_layer=num_layer, bidir=bidir)
    path = os.path.join(out_dir('ckpt_sim'), f'{method_name}_{train_city}_{ldd.nTrain}-{ldd.nTest}_d{dim}n{num_layer}b{(1 if bidir else 0)}_d0.1n0.0002_l{ts_max_len}{ts_min_len}b{batch_size}e{epoch_min}s{earlystop}.th')
    if os.path.exists(path): load_weight(model, path)
    return (model, path)

def infer_noise(method_name, train_city, to_city, TS, dim,drop_rate,noise_rate):
    """load(model,train_city), adapt(to_city), return model(ts)"""
    global augment_drop_rate,augment_noise_rate
    augment_drop_rate,augment_noise_rate=drop_rate,noise_rate
    model, path = _trained_model_path(method_name, train_city,dim)
    assert os.path.exists(path), 'Model untrained:'+path
    model.eval()
    model.to(device)
    if 'GPE' not in method_name:
        TS = migrate_space(train_city, to_city, TS)
    res = []
    with torch.no_grad():
        for ts, les, masks in DataLoader(BaseTrajDataset(to_city, TS, train=False, do_noise=True), batch_size=batch_size, shuffle=False, drop_last=False):
            ts, masks = (cut_len(ts, les), cut_len(masks, les))
            ts, masks = (ts.to(device), masks.to(device))
            vs = model(ts, masks).detach().cpu()
            res.append(vs)
    model.cpu()
    res = torch.concat(res, dim=0)
    return res




from _tool.mIO import mcache
from _data.vector.sim_task import rankvs
device = 'cpu'
task_dir = 'tasks_noise'

@mcache(redir=out_dir(task_dir))
def infer_rank_split(method_name, from_data, to_data, dim,ntr, nte,drop,noise):
    TA, TB = rank_split_data(to_data, nte)
    VA = infer_noise(method_name, from_data, to_data, TA,dim,drop,noise)
    VB = infer_noise(method_name, from_data, to_data, TB,dim,drop,noise)
    return (VA, VB)

@mcache(redir=out_dir(task_dir))
def migrate_mrr(method_name, from_data, to_data, dim,ntr, nte,drop,noise):
    VA, VB = infer_rank_split(method_name, from_data, to_data, dim,ntr, nte,drop,noise)
    VA, VB = (VA.to(device), VB.to(device))
    ranks = rankvs(VA, VB)
    mrr= (1/ranks).mean(dtype=float).item()
    return mrr





fun_base = ['Grid', 'SINW', 'XY', 'XYVV', 'XYDD', 'XYRLG', 'XYG', 'XYSARD', 'Gxy']
fun_gpe = ['GPE_1e-06'] 
dimD=128 
tb_span = ' & '

def run_noise(drop_rates,noise_rates,fun_all=fun_base + fun_gpe,datas=datasets,metric='MRR'):
    print('noise on',datas, metric, ' drop',drop_rates,' noise',noise_rates)
    for ri,rate in enumerate(drop_rates):
        for fi,fun in enumerate(fun_all):
            for di,city in enumerate(datas):
                val=migrate_mrr(fun,city,city,dimD,ldd.nTrain,ldd.nTest,rate,0)
                print(fun,city,'drop',rate,val)
    for ri,rate in enumerate(noise_rates):
        for fi,fun in enumerate(fun_all):
            for di,city in enumerate(datas):
                val=migrate_mrr(fun,city,city,dimD,ldd.nTrain,ldd.nTest,0,rate)
                print(fun,city,'noise',rate,val)
    
if __name__ == '__main__':
    drop_rates=[0.1,0.2,0.3,0.4,0.5]
    noise_rates=[0.1,0.2,0.3,0.4,0.5]
    run_noise(drop_rates,noise_rates,datas=['T-drive'])