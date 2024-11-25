import sys,time,torch
import numpy as np
sys.path.extend(['./', '../../', '../'])
from _tool.SysMonitor import GroupLogS, pttb_value_rank
from _tool.mFile import log_dir
from traj.GPE.GPE import gpe_pretrain
from traj.GPE.downtask import migrate_rank, migrate_prec, migrate_knn_self,migrate_mrr
from traj.GPE.model_sim import train_sim
from _data.traj.load_trajs import traj_bbox
from traj.GPE.gridE import GridEbd, GxyEbd
from traj.GPE.loaddata import dxy, datasets, rank_split_data
import traj.GPE.loaddata as ldd
print = GroupLogS(log_dir())
print.set('GPE.main')
fun_base = ['Grid', 'SINW', 'XY', 'XYVV', 'XYDD', 'XYRLG', 'XYG', 'XYSARD', 'Gxy']
fun_gpe = ['GPE_1e-06', 'GPEwT_1e-06_0.0001']
metrics = {'MeanRank': migrate_rank, 'MRR':migrate_mrr,'MeanPrec': migrate_prec, 'knnSelf10': lambda x1, x2, x3, x4, x5: migrate_knn_self(x1, x2, x3, x4, x5, 10),} 
metri_rw = {'MeanRank': 1, 'MeanPrec': -1, 'knnSelf10': -1,'MRR':-1} # -1: bigger better
tb_span = ' & '

def init_cpu():
    """prepare data, run before real experiments"""
    fbox = lambda b: [b[0][1] - b[0][0], b[1][1] - b[1][0]]
    for city in datasets:
        print('process dataset', city)
        rank_split_data(city, ldd.nTest)
        bbox = traj_bbox[city]
        print('city:', city, ' bbox:', bbox, ' range:', fbox(bbox))
    dim = 128
    for city in datasets:
        for d in [dim, dim // 2]:
            g = GridEbd(d, city, dxy=dxy(city))
            print(f'{city} num_grids:{g._space.num_grids}')
            print('GridEbd word2vec', city, dim)
            g.word2vec()

def init_gpu():
    dim = 128
    for g in fun_gpe:
        print('pretrain', g)
        gpe_pretrain(g, dim)
    for city in datasets:
        for dim in [64, 128]:
            g = GxyEbd(dim, city, dxy=dxy(city))
            print('Gxy pretrain', city, dim)
            g.pretrain()
    for fun in fun_gpe + fun_base:
        for city in datasets:
            train_sim(fun, city)
            for mi, metric in enumerate(metrics):
                evafun = metrics[metric]
                for city2 in datasets:
                    val = evafun(fun, city, city2, ldd.nTrain, ldd.nTest)

def replace_fun_name(name):
    d = {'SINW': 'TriW', 'Gxy': 'GXGY','GPE_1e-06':'GPE','GPEwT_1e-06_0.0001':'GPEwT'}
    return d[name] if name in d else name

def runtb_rank_knn_cs(fun_sim_all=fun_base + fun_gpe, datasets=datasets, dorank=False, latex12=True):
    print('runtb_rank_knn_cs on nTrain', ldd.nTrain, 'nTest', ldd.nTest)
    nTs = ['all']
    data = np.zeros((len(metrics), len(nTs), len(datasets), len(fun_sim_all)))
    dim_cols = [list(metrics), [str(x) for x in nTs], datasets, fun_sim_all]
    for ni, nT in enumerate(nTs):
        for mi, metric in enumerate(metrics):
            evafun = metrics[metric]
            for di, city in enumerate(datasets):
                for fi, method in enumerate(fun_sim_all):
                    train_sim(method, city)
                    val = evafun(method, city, city, nT, ldd.nTest)
                    data[mi, ni, di, fi] = val
    funnames = [replace_fun_name(x) for x in fun_sim_all]
    rw = [metri_rw[k] for k in metrics]
    for mi in range(len(rw)):
        pttb_value_rank(data[mi, 0, :, :].T, row_names=funnames, col_names=dim_cols[2], col_rw=[rw[mi] for _ in datasets], title=f'{dim_cols[0][mi]}, funs v.s. data', round_len=3, sum_rank=False, tb_span=tb_span, dosum=False, dorank=dorank, latex12=latex12)
    return data

def runtb_cross(fun_sim_all=fun_base + fun_gpe, from_datas=datasets, dorank=False, latex12=True, metric='MeanRank'):
    print('runtb_cross on nTrain', metric, ldd.nTrain, 'nTest', ldd.nTest)
    data = np.zeros((len(from_datas), len(datasets), len(fun_sim_all)))
    evfun = metrics[metric]
    for d1, from_data in enumerate(from_datas):
        for fi, method in enumerate(fun_sim_all):
            train_sim(method, from_data)
            for d2, to_data in enumerate(datasets):
                val = evfun(method, from_data, to_data, ldd.nTrain, ldd.nTest)
                data[d1, d2, fi] = val
    funnames = [replace_fun_name(x) for x in fun_sim_all]
    for d1, from_data in enumerate(from_datas):
        d2s = np.array([1 if d1 != d2 else 0 for d2 in range(len(datasets))], dtype=bool)
        dat = data[d1, d2s].T
        col_names = [x for d2, x in enumerate(datasets) if d1 != d2]
        pttb_value_rank(dat, row_names=funnames, col_names=col_names, col_rw=metri_rw[metric], title=f'cross from {from_data} on {metric}', round_len=3, tb_span=tb_span, dorank=dorank, sum_rank=False, latex12=latex12)
    return data

def runtb_ana():
    print.set('runtb_ana')
    pttbkw = {'dorank': True, 'latex12': False}
    e1d, e2d=1e-06,0.0001
	
    print('\n\n\n\n\n\n E1 \n\n\n\n\n\n')
    e1s = [1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1] 
    name_GPEs = [f'GPE_{str(e1)}' for e1 in e1s]
    data = runtb_rank_knn_cs(name_GPEs, datasets)
    d2= np.mean(data,axis=2) # data[:, 0, 0, :].T
    pttb_value_rank(d2[:, 0, :].T, row_names=name_GPEs, col_names=list(metrics.keys()), col_rw=[metri_rw[mi] for mi in metrics], title=f'GPEs v.s. metrics', round_len=3, sum_rank=True, tb_span=tb_span, dosum=True, dorank=True, latex12=False)
    # # for metric in metrics: runtb_cross(name_GPEs, datasets, metric=metric, **pttbkw)
    # runtb_cross(name_GPEs, datasets, **pttbkw)

    print('\n\n\n\n\n\n E2 \n\n\n\n\n\n')
    e2s = [1e-08,1e-07, 1e-06, 1e-05, 0.0001, 0.001,0.01, 0.1]
    name_wTs = [f'GPEwT_{str(e1d)}_{str(e2)}' for  e2 in e2s]
    for name in name_wTs: gpe_pretrain(name=name)
    data = runtb_rank_knn_cs(name_wTs, datasets) # np.zeros((len(metrics), len(nTs), len(datasets), len(fum)))
    d2=np.mean(data,axis=2)
    pttb_value_rank(d2[:, 0, :].T, row_names=name_wTs, col_names=list(metrics.keys()), col_rw=[metri_rw[mi] for mi in metrics], title=f'GPEs v.s. metrics', round_len=3, sum_rank=True, tb_span=tb_span, dosum=True, dorank=True, latex12=False)
    # runtb_cross(name_wTs, datasets, **pttbkw)
    # for mi, metric in enumerate(metrics):pttb_value_rank(data[mi, 0, :, :].T, row_names=name_wTs, col_names=datasets, col_rw=metri_rw[metric], title=f'GPEs v.s. ' + metric, round_len=3, sum_rank=True, tb_span=tb_span, dosum=True, dorank=True, latex12=False)

    print('\n\n\n\n\n\n early stop \n\n\n\n\n\n')
    namesT = ['GPEwT_1e-06_0.0001_1b','GPEwT_1e-06_0.0001_1k','GPEwT_1e-06_0.0001']
    for name in namesT: gpe_pretrain(name=name)
    d2= np.mean(data,axis=2) # data[:, 0, 0, :].T
    pttb_value_rank(d2[:, 0, :].T, row_names=namesT, col_names=list(metrics.keys()), col_rw=[metri_rw[mi] for mi in metrics], title=f'GPEs v.s. estop', round_len=3, sum_rank=True, tb_span=tb_span, dosum=True, dorank=True, latex12=False)

    print.unset()

if __name__ == '__main__':
    init_cpu()
    init_gpu()
    runtb_rank_knn_cs()
    runtb_cross(metric='MeanRank')
    runtb_ana()
