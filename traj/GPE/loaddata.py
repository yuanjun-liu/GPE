import sys
import numpy as np
sys.path.extend(['./', '../../', '../'])
from _tool.mIO import mcache
from _data.traj.process_ts import ts_split_2zip, ts_split_step
from _data.traj.load_trajs import load_ts_box, traj_bbox
cellsize = 0.0015
cellsize_ais = 0.31484
dxy = lambda city: (cellsize_ais, cellsize_ais) if city == 'AIS' else (cellsize, cellsize)
ts_max_len = 200
ts_min_len = 20
nTest = 5000
nTrain = 'all'
datasets = ['T-drive', 'Porto', 'Roma', 'AIS', 'GeoLife']

@mcache()
def loadss(data, len_max, len_min):
    TS = load_ts_box(data)[0]
    if len(TS) < 6:
        TS = TS[0]
    TS = ts_split_step(TS, len_max, len_min, len_max // 2)
    return TS

def loadt2(data: str, ntrain: int, ntest: int):
    TS = loadss(data, ts_max_len, ts_min_len)
    if ntrain == 'all':
        ntrain = -ntest
    TS_train, TS_test = (TS[:ntrain], TS[-ntest:])
    return (TS_train, TS_test)

def load_traj(data, ntest=nTest):
    return loadt2(data, nTrain, ntest)
traj_mean_len = {'T-drive': 136, 'Porto': 53, 'Roma': 183, 'AIS': 39,  'GeoLife': 185, }

def migrate_space(city_old, city_new, ts):
    """only cross city
    new_p=(old_p-old_min)/(old_max-old_min)*(new_max-new_min)+new_min"""
    if city_old == city_new:
        return ts
    bbox_new, bbox_old = (traj_bbox[city_new], traj_bbox[city_old])
    old_min, new_min = (np.array(bbox_old[0]), np.array(bbox_new[0]))
    old_range = np.array([bbox_old[0][1] - bbox_old[0][0], bbox_old[1][1] - bbox_old[1][0]])
    new_range = np.array([bbox_new[0][1] - bbox_new[0][0], bbox_new[1][1] - bbox_new[1][0]])
    TS = []
    if len(ts) < 5:
        ts = ts[0]
    for t in ts:
        t = np.array(t)
        t[:, :2] = (t[:, :2] - old_min) / old_range * new_range + new_min
        TS.append(t)
    return TS

@mcache()
def rank_split_data(data, n):
    ts = load_traj(data, n)[1]
    ta, tb = ts_split_2zip(ts)
    return (ta, tb)