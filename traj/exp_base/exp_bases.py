import sys
import numpy as np
sys.path.extend(['./', '../../', '../'])
from _tool.SysMonitor import  pttb_value_rank
from traj.exp_base.task_bs import migrate_rank, migrate_prec, migrate_knn_self,migrate_mrr
from traj.exp_base.model_bs import train_bs
from traj.exp_base.GPE_bases import gpe_bases_info
from traj.GPE.loaddata import  datasets
import traj.GPE.loaddata as ldd
dimD=128 
metrics = {'MeanRank': migrate_rank, 'MRR':migrate_mrr,'MeanPrec': migrate_prec, 'knnSelf10': lambda x1, x2, x3,dim1,dim2, x4, x5: migrate_knn_self(x1, x2, x3,dim1,dim2, x4, x5, 10),} 
metri_rw = {'MeanRank': 1, 'MeanPrec': -1, 'knnSelf10': -1,'MRR':-1}
tb_span = ' & '


def runtb_bases(methods,dim1=dimD):
    data = np.zeros((len(metrics), len(datasets), len(methods)))
    for fi, fun in enumerate(methods):
        for d1,city1 in enumerate(datasets):
            train_bs(fun,city1,dim1,dimD)
            for mi,metric in enumerate(metrics):
                evafun = metrics[metric]
                val=evafun(fun,city1,city1,dim1,dimD,'all',ldd.nTest)
                data[mi,d1,fi]=val
    d2= np.mean(data,axis=1) 
    pttb_value_rank(d2.T, row_names=methods, col_names=list(metrics.keys()), col_rw=[metri_rw[mi] for mi in metrics], title=f'bases', round_len=3, sum_rank=True, tb_span=tb_span, dosum=True, dorank=True, latex12=False)


if __name__ == '__main__':
    pass
    dim1=256 
    methods=['bGPEInc_1e-6','bGPEPrime_1e-6','bGPE_1e-6','bGPEAtt_1e-6','bGPESingle']

    print('base info')
    for fun in methods:
        print(fun,gpe_bases_info(fun,dim1))

    runtb_bases(methods,dim1=dim1) 
