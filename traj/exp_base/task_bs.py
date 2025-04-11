import sys
import torch
import numpy as np
sys.path.extend(['./', '../../', '../'])
from _tool.mIO import mcache
from _tool.mFile import out_dir
from _data.vector.sim_task import rankvs, knnvd, knn_prec
from traj.exp_base.model_bs import infer_bs
from traj.GPE.loaddata import rank_split_data, load_traj
device = 'cpu'
task_dir = 'tasks_bs'

def prec_rank(rks: torch.Tensor):
    return (rks == 1).sum() / len(rks)

@mcache(redir=out_dir(task_dir))
def infer_rank_split(method_name, from_data, to_data, dim1,dim2,ntr, nte):
    TA, TB = rank_split_data(to_data, nte)
    VA = infer_bs(method_name, from_data, to_data, TA,dim1,dim2)
    VB = infer_bs(method_name, from_data, to_data, TB,dim1,dim2)
    return (VA, VB)

@mcache(redir=out_dir(task_dir))
def migrate_rank(method_name, from_data, to_data,dim1,dim2, ntr, nte):
    VA, VB = infer_rank_split(method_name, from_data, to_data, dim1,dim2,ntr, nte)
    VA, VB = (VA.to(device), VB.to(device))
    mean_rank = rankvs(VA, VB).mean(dtype=float).item()
    return mean_rank

@mcache(redir=out_dir(task_dir))
def migrate_mrr(method_name, from_data, to_data, dim1,dim2,ntr, nte):
    VA, VB = infer_rank_split(method_name, from_data, to_data, dim1,dim2,ntr, nte)
    VA, VB = (VA.to(device), VB.to(device))
    ranks = rankvs(VA, VB)
    mrr= (1/ranks).mean(dtype=float).item()
    return mrr

@mcache(redir=out_dir(task_dir))
def migrate_prec(method_name, from_data, to_data, dim1,dim2,ntr, nte):
    VA, VB = infer_rank_split(method_name, from_data, to_data,dim1,dim2, ntr, nte)
    VA, VB = (VA.to(device), VB.to(device))
    mean_rank = prec_rank(rankvs(VA, VB))
    return mean_rank

@mcache(redir=out_dir(task_dir))
def migrate_knn_self(method_name, from_data, to_data, dim1,dim2,ntr, nte, k):
    VQ1, VD1 = infer_rank_split(method_name, from_data, to_data,dim1,dim2, ntr, nte)
    VQ1, VD1 = (VQ1.to(device), VD1.to(device))
    K1 = knnvd(VQ1, VD1, k)
    ts = load_traj(to_data, nte)[1]
    VT = infer_bs(method_name, from_data, to_data, ts,dim1,dim2)
    K2 = knnvd(VT, VT, k)
    assert len(K1) == len(K2)
    K1, K2 = (K1.to(device), K2.to(device))
    kp = knn_prec(K1, K2).item()
    return kp