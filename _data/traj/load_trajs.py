import os, sys
import numpy as np
sys.path.extend(['./', '../', '../../'])
from _tool.mList import choice, choice_co
from _tool.mData import dicted_idx
from _tool.mFile import path_traj, list_dir, read_lines_iter, parse_line, out_base
from _tool.mIO import mcache
from _tool.mTime import tim2sec
from _tool.SysMonitor import print_table
from _data.traj.process_ts import ts_len_info, ts_bound, ts_shrink

@mcache()
def sample(name: str or list or np.ndarray, num):
    Ts = load(name) if isinstance(name, str) else name
    le = len(Ts[0]) if len(Ts) == 2 else len(Ts)
    if len(Ts) == 2:
        return choice_co(Ts, num, num > le)
    else:
        return choice(Ts, num, num > le)
traj_data_path = {'T-drive': os.path.join(path_traj, 'T-drive/taxi_log_2008_by_id'), 'GeoLife': os.path.join(path_traj, 'GeoLife'), 'Porto': os.path.join(path_traj, 'Porto/train.csv'), 'Roma': os.path.join(path_traj, 'Roma.txt'), 'AIS': os.path.join(path_traj, 'AIS'),}
ts_data_name = list(traj_data_path.keys())
traj_mean_len = {'T-drive': 203, 'GeoLife': 1357, 'Porto': 50, 'Roma': 212,  'AIS': 45}
filter_len = 10

def load_Tdrive(name='T-drive', time_span='30m', filter_len=filter_len):
    assert name in traj_data_path
    data = traj_data_path[name]
    time_span = tim2sec(time_span)
    _tim = 0
    trajs = []
    T = []
    _file = ''
    for file in list_dir(data)[3]:
        for line in read_lines_iter(file):
            uid_, tim, lon, lat = parse_line(line, ',', int, str, float, float)
            tim = tim2sec(tim)
            if _tim == 0:
                _tim = tim
                _file = file
            if abs(tim - _tim) > time_span or _file != file:
                if len(T) > filter_len:
                    trajs.append(np.array(T))
                T = []
                _file = file
            _tim = tim
            T.append([lat, lon, tim])
        if len(T) > filter_len:
            trajs.append(np.array(T))
    return np.array(trajs, dtype=object)

def load_GeoLife(name='GeoLife', filter_len=filter_len):
    assert name in traj_data_path
    data = traj_data_path[name]
    trajs = []
    _file = ''
    for uid in list_dir(data)[0]:
        dir = os.path.join(data, uid, 'Trajectory')
        for file in list_dir(dir)[3]:
            T = []
            for line in read_lines_iter(file, 6):
                lat, lon, _, _, _, t1, t2 = parse_line(line, ',', float, float, str)
                tim = tim2sec(t1 + ' ' + t2)
                T.append([lat, lon, tim])
            if len(T) > filter_len:
                trajs.append(np.array(T))
    return np.array(trajs, dtype=object)

def load_AIS(name='AIS', time_span='1h', filter_len=filter_len):
    assert name in traj_data_path
    data = traj_data_path[name]
    time_span = tim2sec(time_span)
    trajs = []
    T = dict()
    _t = dict()
    for file in list_dir(data)[3]:
        for line in read_lines_iter(file, 1):
            uid, tim, lat, lon, *other = parse_line(line, ',', str, str, float, float, str)
            uid = dicted_idx(name, uid)
            tim = ' '.join(tim.split('T'))
            tim = tim2sec(tim)
            if uid not in T:
                T[uid] = []
            if uid not in _t:
                _t[uid] = tim
            if abs(tim - _t[uid]) > time_span:
                if len(T[uid]) > filter_len:
                    trajs.append(np.array(T[uid]))
                T[uid] = []
            _t[uid] = tim
            T[uid].append([lat, lon, tim])
        break
    for t in T:
        if len(T[t]) > filter_len:
            trajs.append(np.array(T[t]))
    return np.array(trajs, dtype=object)

def load_Roma(name='Roma', time_duration='1m', filter_len=filter_len):
    assert name in traj_data_path
    data = traj_data_path[name]
    time_span = tim2sec(time_duration)
    trajs = []
    T = dict()
    _t = dict()
    for line in read_lines_iter(data):
        uid, tim, point = parse_line(line, ';', int, str, str)
        lat, lon = parse_line(point[6:-1], ' ', float, float)
        uid = dicted_idx(name, uid)
        tim = tim2sec(tim[:19])
        if uid not in T:
            T[uid] = []
        if uid not in _t:
            _t[uid] = tim
        if abs(tim - _t[uid]) > time_span:
            if len(T[uid]) > filter_len:
                trajs.append(np.array(T[uid]))
            T[uid] = []
        _t[uid] = tim
        T[uid].append([lat, lon, tim])
    for t in T:
        if len(T[t]) > filter_len:
            trajs.append(np.array(T[t]))
    return np.array(trajs, dtype=object)


def load_Porto(name='Porto', filter_len=filter_len):
    assert name in traj_data_path
    data = traj_data_path[name]
    _tim = 0
    trajs = []
    for line in read_lines_iter(data, 1):
        T = []
        points = line.split('"')[-2][2:-3]
        if points == '':
            continue
        for point in points.split('],['):
            pp = point.split(',')
            lat, lon = (float(pp[0]), float(pp[1]))
            T.append([lat, lon, len(T)])
        if len(T) > filter_len:
            trajs.append(np.array(T))
    return np.array(trajs, dtype=object)

_load_funs = {'T-drive': load_Tdrive, 'GeoLife': load_GeoLife, 'Porto': load_Porto, 'Roma': load_Roma,'AIS':load_AIS}

def load(data):
    redir = os.path.join(out_base, 'TrajData')
    assert data in _load_funs
    fun = _load_funs[data]
    cache_name = f'load({'_'.join([str(x) for x in fun.__defaults__])})'
    data = mcache(cache_name, redir=redir)
    if data is not None:
        return data
    ts = fun()
    mcache(cache_name, ts, redir=redir)
    return ts

def info_len(data):
    assert data in _load_funs
    fun = _load_funs[data]
    cache_name = f'infoLen({'_'.join([str(x) for x in fun.__defaults__])})'
    res = mcache(cache_name, ftype='json')
    if res is not None:
        return res
    traj = load(data)
    if len(traj) == 2:
        traj = traj[0]
    res = ts_len_info(traj)
    mcache(cache_name, res, ftype='json')
    return res

def info_len_table():
    infoS = [['traj', 'num', 'len_max', 'len_min', 'len_avg']]
    for data in ts_data_name:
        info = info_len(data)
        infoS.append([data, info['num'], info['len_max'], info['len_min'], info['len_avg']])
    print_table(infoS[1:], infoS[0])
traj_bbox = {'T-drive': [[39.8, 40.05], [116.25, 116.5]], 'GeoLife': [[39.8, 40.05], [116.25, 116.5]],  'Porto': [[-8.69043114799999, -8.51008941766], [41.086125252, 41.255937432]], 'Roma': [[41.79027231408, 41.962425621120005], [12.251662229770494, 12.588559427058325]],  'AIS': [[18.296277252, 49.292036766], [-156.490691348, -65.30201888300002]]}

@mcache(name='loadB', redir=os.path.join(out_base, 'TrajData'))
def load_ts_box(data):
    assert data in traj_bbox
    ts = load(data)
    if len(ts) < 4:
        ts = ts[0]
    if len(ts) < 4:
        ts = ts[0]
    if traj_bbox[data] is None:
        ts, bbox = ts_shrink(ts, 0.01)
    else:
        bbox = traj_bbox[data]
        ts = ts_bound(ts, traj_bbox[data])
    return (ts, bbox)
