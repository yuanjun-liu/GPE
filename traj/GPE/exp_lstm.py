import sys
import numpy as np
sys.path.extend(['./', '../../', '../'])
from _tool.SysMonitor import GroupLogS, pttb_value_rank,LogToTxt
from _tool.mFile import log_dir
from traj.GPE.task_lstm import migrate_rank, migrate_prec, migrate_knn_self,migrate_mrr,infer_rank_split
from traj.GPE.model_lstm import train_lstm
from _data.traj.load_trajs import traj_bbox
from traj.GPE.gridE import GridEbd, GxyEbd
from traj.GPE.loaddata import dxy, datasets, rank_split_data
import traj.GPE.loaddata as ldd
print = GroupLogS(log_dir(), cls=LogToTxt) 
print.set('GPE.main')
fun_base = ['Grid', 'SINW', 'XY', 'XYVV', 'XYDD', 'XYRLG', 'XYG', 'XYSARD', 'Gxy']
fun_gpe = ['GPE_1e-06'] 
dimS=[32,64,128,256] ; dimD=128 
metrics = {'MeanRank': migrate_rank, 'MRR':migrate_mrr,'MeanPrec': migrate_prec, 'knnSelf10': lambda x1, x2, x3,dim, x4, x5: migrate_knn_self(x1, x2, x3,dim, x4, x5, 10),} 
metri_rw = {'MeanRank': 1, 'MeanPrec': -1, 'knnSelf10': -1,'MRR':-1}
tb_span = ' & '

def init_cpu():
    """prepare data, run before real experiments"""
    fbox = lambda b: [b[0][1] - b[0][0], b[1][1] - b[1][0]]
    for city in datasets:
        print('process dataset', city)
        rank_split_data(city, ldd.nTest)
        bbox = traj_bbox[city]
        print('city:', city, ' bbox:', bbox, ' range:', fbox(bbox))

def init_gpu(dim,funs=fun_gpe + fun_base,citys=datasets):
    for city in citys:
        for d in [dim, dim//2]:
            g = GridEbd(d, city, dxy=dxy(city))
            if not g._trained:
                print(f'{city} num_grids:{g._space.num_grids}')
                print('GridEbd word2vec', city, d)
                g.word2vec()
                print('over')
            g = GxyEbd(d, city, dxy=dxy(city))
            if not g._trained:
                print('GxGy pretrain', city, d)
                g.pretrain()
                print('over')
    for fun in funs:
        print('init_gpu fun',fun)
        for city in citys:
            print('init_gpu city',city)
            train_lstm(fun, city,dim)
            for mi, metric in enumerate(metrics):
                evafun = metrics[metric]
                for city2 in datasets:
                    infer_rank_split(fun,city,city2,dim,ldd.nTrain, ldd.nTest)
                    val = evafun(fun, city, city2, dim,ldd.nTrain, ldd.nTest)

def replace_fun_name(name):
    d = {'SINW': 'TriW', 'Gxy': 'GXGY','GPE_1e-06':'GPE'}
    return d[name] if name in d else name

def runtb_local(dim,fun_sim_all=fun_base + fun_gpe, datasets=datasets, dorank=False, latex12=True):
    print(f'runtb_rank_knn_cs d{dim} on nTrain', ldd.nTrain, 'nTest', ldd.nTest)
    nTs = ['all']
    data = np.zeros((len(metrics), len(nTs), len(datasets), len(fun_sim_all)))
    dim_cols = [list(metrics), [str(x) for x in nTs], datasets, fun_sim_all]
    for ni, nT in enumerate(nTs):
        for mi, metric in enumerate(metrics):
            evafun = metrics[metric]
            for di, city in enumerate(datasets):
                for fi, method in enumerate(fun_sim_all):
                    train_lstm(method, city,dim)
                    val = evafun(method, city, city, dim,nT, ldd.nTest)
                    data[mi, ni, di, fi] = val
    funnames = [replace_fun_name(x) for x in fun_sim_all]
    rw = [metri_rw[k] for k in metrics]
    for mi in range(len(rw)):
        pttb_value_rank(data[mi, 0, :, :].T, row_names=funnames, col_names=dim_cols[2], col_rw=[rw[mi] for _ in datasets], title=f'{dim_cols[0][mi]}, funs v.s. data', round_len=3, sum_rank=False, tb_span=tb_span, dosum=True, dorank=dorank, latex12=latex12)
    return data

def runtb_h(fun_sim_all=fun_base + fun_gpe):
    print(f'runtb_dims on nTrain', ldd.nTrain, 'nTest', ldd.nTest)
    datasets=['T-drive']
    metrics={'MRR':migrate_mrr} ; round_len=3
    data = np.zeros((len(metrics), len(dimS), len(datasets), len(fun_sim_all)))
    dim_cols = [list(metrics), [str(d) for d in dimS], datasets, fun_sim_all]
    for dii, dim in enumerate(dimS):
        for mi, metric in enumerate(metrics):
            evafun = metrics[metric]
            for dai, city in enumerate(datasets):
                for fi, method in enumerate(fun_sim_all):
                    train_lstm(method, city,dim)
                    val = evafun(method, city, city, dim,'all', ldd.nTest)
                    data[mi, dii, dai, fi] = val
    funnames = [replace_fun_name(x) for x in fun_sim_all]
    rw = [metri_rw[k] for k in metrics]
    for mi in range(len(rw)):
        pttb_value_rank(data[mi, :, 0, :].T, row_names=funnames, col_names=dim_cols[1], col_rw=rw[0], title=f'funs v.s. h', round_len=round_len, sum_rank=False, tb_span=tb_span, dosum=False, dorank=False, latex12=True)
    return data


def runtb_global(dim,fun_sim_all=fun_base + fun_gpe, from_datas=datasets, dorank=False, latex12=True, metric='MRR'):
    print('runtb_cross on nTrain', metric, ldd.nTrain, 'nTest', ldd.nTest)
    data = np.zeros((len(from_datas), len(datasets), len(fun_sim_all)))
    evfun = metrics[metric]
    for d1, from_data in enumerate(from_datas):
        for fi, method in enumerate(fun_sim_all):
            train_lstm(method, from_data,dim)
            for d2, to_data in enumerate(datasets):
                val = evfun(method, from_data, to_data,dim, ldd.nTrain, ldd.nTest)
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

    e1s = [1e-08, 1e-07, 1e-06, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1] # 1e-09 ,2.5e-06
    name_GPEs = [f'GPE_{str(e1)}' for e1 in e1s]
    data = runtb_local(dimD,name_GPEs, datasets)
    d2= np.mean(data,axis=2) # data[:, 0, 0, :].T
    pttb_value_rank(d2[:, 0, :].T, row_names=name_GPEs, col_names=list(metrics.keys()), col_rw=[metri_rw[mi] for mi in metrics], title=f'GPEs v.s. metrics', round_len=3, sum_rank=True, tb_span=tb_span, dosum=True, dorank=True, latex12=False)
    # # for metric in metrics: runtb_cross(name_GPEs, datasets, metric=metric, **pttbkw)
    # runtb_cross(name_GPEs, datasets, **pttbkw)

if __name__ == '__main__':
    pass
    init_cpu()
    init_gpu(dimD)
    runtb_local(dimD)
    runtb_global(dimD,metric='MRR')
    runtb_ana() # for para e of GPE

    # embedding size h
    for d in dimS:  init_gpu(d,citys=['T-drive'])
    runtb_h()
    
    # for time, check logs
    