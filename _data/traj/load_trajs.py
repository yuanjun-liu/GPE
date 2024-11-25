import os, sys
import numpy as np
sys.path.extend(['./', '../', '../../'])
from _tool.mList import choice, choice_co
from _tool.mData import dicted_idx
from _tool.mFile import path_traj, list_dir, read_lines_iter, parse_line, out_base
from _tool.mIO import mcache
from _tool.mTime import tim2sec
from _tool.SysMonitor import print_table
from _data.traj.process_ts import ts_filter_len, ts_len_info, ts_bound, ts_shrink

@mcache()
def sample(name: str or list or np.ndarray, num):
    Ts = load(name) if isinstance(name, str) else name
    le = len(Ts[0]) if len(Ts) == 2 else len(Ts)
    if len(Ts) == 2:
        return choice_co(Ts, num, num > le)
    else:
        return choice(Ts, num, num > le)
traj_data_path = {'T-drive': os.path.join(path_traj, 'T-drive/taxi_log_2008_by_id'), 'GeoLife': os.path.join(path_traj, 'GeoLife'), 'GeoLifeLabel': os.path.join(path_traj, 'GPS Trajectories with transportation mode labels'), 'MC': os.path.join(path_traj, 'MobileCentury_data_final_ver3/'), 'Porto': os.path.join(path_traj, 'Porto/train.csv'), 'Roma': os.path.join(path_traj, 'Roma.txt'), 'SFBA': os.path.join(path_traj, 'SanFranciscoBayArea'), 'AIS': os.path.join(path_traj, 'AIS'), 'ASL2': os.path.join(path_traj, 'ASL2'), 'TaxiSH': os.path.join(path_traj, 'TaxiSH/TaxiSH'), 'vanet-trace': os.path.join(path_traj, 'vanet-trace-creteil')}
ts_data_name = list(traj_data_path.keys())
traj_mean_len = {'T-drive': 203, 'GeoLife': 1357, 'GeoLifeLabel': 523, 'TaxiSH': 97, 'MC': 307, 'Porto': 50, 'Roma': 212, 'SFBA': 20, 'AIS': 45, 'ASL2': 57, 'vanet-trace': 366}
filter_len = 10

def load_HumanFootv2():
    ...

def load_vanet_trace(name='vanet-trace'):
    assert name in traj_data_path
    dir = traj_data_path[name]
    ts = {}
    tps = {}
    for fi, file in enumerate(['vanet-trace-creteil-20130924-0700-0900.csv', 'vanet-trace-creteil-20130924-1700-1900.csv']):
        for line in read_lines_iter(os.path.join(dir, file), 1):
            if ';;' in line:
                continue
            tim, _, _, ang, typ, pos, y, x, sped, carid = parse_line(line, ';', float, str, str, float, str, str, float, float, float, str)
            i = dicted_idx(f'id{fi}', carid)
            if i not in ts:
                ts[i] = []
            ts[i].append([x, y, tim, ang, sped])
            tps[i] = dicted_idx('type', typ)
    ks = list(ts.keys())
    i = 0
    while i < len(ks):
        if len(ts[i]) < filter_len:
            del ks[i], ts[i], tps[i]
        else:
            i += 1
    trajs = [ts[k] for k in ks]
    typs = [tps[k] for k in ks]
    return (trajs, typs)

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

def load_TaxiSH(name='TaxiSH', time_span='10m', filter_len=filter_len):
    assert name in traj_data_path
    data = traj_data_path[name]
    time_span = tim2sec(time_span)
    trajs = []
    T = []
    _tim = 0
    _file = ''
    _on = 0
    for file in list_dir(data)[3]:
        if 'readme' in file:
            continue
        for line in read_lines_iter(file):
            if line == '':
                continue
            _, tim, lon, lat, speed, angle, oncar = parse_line(line, ',', int, str, float, float, int, int, int)
            tim = tim2sec(tim)
            if _tim == 0:
                _tim = tim
                _on = oncar
                _file = file
            if abs(tim - _tim) > time_span or _file != file or _on != oncar:
                if len(T) > filter_len:
                    trajs.append(np.array(T))
                T = []
                _file = file
                _on = oncar
            _tim = tim
            T.append([lat, lon, tim])
        if len(T) > filter_len:
            trajs.append(np.array(T))
    return np.array(trajs, dtype=object)

def muban(name, time_span, filter_len=filter_len):
    assert name in traj_data_path
    data = traj_data_path[name]
    time_span = tim2sec(time_span)
    _tim = 0
    trajs = []
    T = []
    _file = ''
    for file in list_dir(data):
        for line in read_lines_iter(file):
            tim, lat, lon = parse_line(line, ',', str, float, float)
            tim = tim2sec(tim)
            if _tim == 0:
                _tim = tim
                _file = file
            if abs(tim - _tim) > time_span or _file != file:
                if len(T) > 1:
                    trajs.append(np.array(T))
                T = []
                _file = file
            _tim = tim
            T.append([lat, lon, tim])
        if len(T) > 1:
            trajs.append(np.array(T))
    if filter_len > 0:
        trajs = ts_filter_len(trajs, filter_len)
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

def load_GeoLifeLabel(name='GeoLifeLabel', filter_len=filter_len):
    assert name in traj_data_path
    data = traj_data_path[name]
    trajs = []
    for dir in list_dir(data)[2]:
        label = []
        file = os.path.join(dir, 'labels.txt')
        for line in read_lines_iter(file, 1):
            dat, t1, t2, mode = parse_line(line, '\t', str)
            if dat == '' or t1 == '' or t2 == '':
                continue
            if '/' in t1:
                continue
            t1 = tim2sec(dat + ' ' + t1)
            t2 = tim2sec(dat + ' ' + t2)
            label.append([t1, t2])
        label = np.array(label)
        T = []
        idx_lb = -1
        _file = ''
        dir = os.path.join(dir, 'trajectory')
        for file in list_dir(dir)[3]:
            for line in read_lines_iter(file, 6):
                lat, lon, code, _, _, t1, t2 = parse_line(line, ',', float, float, str)
                tim = tim2sec(t1 + ' ' + t2)
                tb = np.where((tim >= label[:, 0]) * (tim <= label[:, 1]))[0]
                if len(tb) == 0:
                    continue
                idx = tb[0]
                if idx != idx_lb or _file != file:
                    if T is not None and len(T) > filter_len:
                        trajs.append(np.array(T))
                    T = []
                    idx_lb = idx
                    _file = file
                T.append([lat, lon, tim])
            if T is not None and len(T) > filter_len:
                trajs.append(np.array(T))
    return np.array(trajs, dtype=object)

def load_ASL2(name='ASL2'):
    assert name in traj_data_path
    data = traj_data_path[name]
    _, _, dirs, _ = list_dir(data)
    trajs, signs = ([], [])
    for dir in dirs:
        _, files, _, file_path = list_dir(dir)
        for file, path in zip(files, file_path):
            sign = file.split('-')[0]
            if sign == 'his_hers':
                sign = 'her'
            signs.append(dicted_idx(name, sign))
            T = []
            for t, line in enumerate(read_lines_iter(path)):
                d = parse_line(line, '\t', float)
                T.append([d[0], d[1], t])
            trajs.append(np.array(T))
    return (np.array(trajs, dtype=object), np.array(signs, dtype=int))

def load_ASL(name='ASL', filter_len=filter_len):
    assert name in traj_data_path
    data = traj_data_path[name]
    trajs, signs = ([], [])
    for dir in list_dir(data)[2]:
        _, files, _, file_path = list_dir(dir)
        for file, path in zip(files, file_path):
            sign = file[:-6]
            signs.append(dicted_idx(name, sign))
            T = []
            for t, line in enumerate(read_lines_iter(path)):
                x, y, z, roll, _, _, f1, f2, f3, f4, _, _, _, _, _ = parse_line(line, ',', float, float, float, float, float, float, float, float, float, float, float, str, str, str, str)
                T.append([x, y, t])
            trajs.append(np.array(T))
    i = 0
    while i < len(trajs):
        if len(trajs[i]) < filter_len:
            del trajs[i]
            del signs[i]
        else:
            i += 1
    return (np.array(trajs, dtype=object), np.array(signs, dtype=int))

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

def load_SFBA(name='SFBA', time_span='10m', filter_len=filter_len):
    assert name in traj_data_path
    data = traj_data_path[name]
    time_span = tim2sec(time_span)
    trajs = []
    T = []
    _tim = 0
    _file = ''
    _on = 0
    for file in list_dir(data)[3]:
        if 'readme' in file:
            continue
        for line in read_lines_iter(file):
            lat, lon, oncar, tim = parse_line(line, ' ', float, float, int, int)
            if _tim == 0:
                _tim = tim
                _on = oncar
                _file = file
            if abs(tim - _tim) > time_span or _file != file or _on != oncar:
                if len(T) > filter_len:
                    trajs.append(np.array(T))
                T = []
                _file = file
                _on = oncar
            _tim = tim
            T.append([lat, lon, tim])
        if len(T) > filter_len:
            trajs.append(np.array(T))
    return np.array(trajs, dtype=object)

def load_MC(name='MC', filter_len=filter_len):
    assert name in traj_data_path
    data = traj_data_path[name]
    _tim = 0
    trajs = []
    T = []
    _file = ''
    for dir in list_dir(data)[2]:
        for file in list_dir(dir)[3]:
            for line in read_lines_iter(file, 1):
                tim, lat, lon, *oth = parse_line(line, ',', int, float, float)
                if _tim == 0:
                    _tim = tim
                    _file = file
                if _file != file:
                    if len(T) > filter_len:
                        trajs.append(np.array(T))
                    T = []
                    _file = file
                _tim = tim
                T.append([lat, lon, int(tim / 1000)])
            if len(T) > filter_len:
                trajs.append(np.array(T))
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

def load_koln(name='koln', time_span='10m', filter_len=filter_len):
    assert name in traj_data_path
    file = traj_data_path[name]
    time_span = tim2sec(time_span)
    _tim = dict()
    ts = dict()
    trajs = []
    for line in read_lines_iter(file):
        try:
            tim, uid, lat, lon, _speed = parse_line(line, ' ', float, int, float, float)
        except:
            continue
        if uid not in ts:
            ts[uid] = []
            _tim[uid] = tim
        if abs(tim - _tim[uid]) > time_span:
            if len(ts[uid]) > filter_len:
                trajs.append(np.array(ts[uid]))
            ts[uid] = []
        _tim[uid] = tim
        ts[uid].append([lat, lon, tim])
    for t in ts:
        if len(ts[t]) > filter_len:
            trajs.append(np.array(ts[t]))
    return np.array(trajs, dtype=object)
_load_funs = {'T-drive': load_Tdrive, 'GeoLife': load_GeoLife, 'GeoLifeLabel': load_GeoLifeLabel, 'TaxiSH': load_TaxiSH, 'MC': load_MC, 'Porto': load_Porto, 'Roma': load_Roma, 'SFBA': load_SFBA, 'AIS': load_AIS, 'ASL': load_ASL, 'ASL2': load_ASL2, 'HumanFootprintv2': load_HumanFootv2, 'vanet-trace': load_vanet_trace}

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
traj_bbox = {'T-drive': [[39.8, 40.05], [116.25, 116.5]], 'GeoLife': [[39.8, 40.05], [116.25, 116.5]], 'GeoLifeLabel': [[39.8, 40.05], [116.25, 116.5]], 'TaxiSH': [[30.9194394, 31.3866438], [121.18825217999999, 121.79940979]], 'MC': [[37.527324186822845, 37.65466590187305], [-122.100216321873, -121.99759676479492]], 'Porto': [[-8.69043114799999, -8.51008941766], [41.086125252, 41.255937432]], 'Roma': [[41.79027231408, 41.962425621120005], [12.251662229770494, 12.588559427058325]], 'SFBA': [[37.566893744, 37.824942992000004], [-122.49022904, -122.231670272]], 'AIS': [[18.296277252, 49.292036766], [-156.490691348, -65.30201888300002]]}

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
if __name__ == '__main__':
    x = load_ts_box('AIS')
    print(len(x[0]))
    print(x[1])
    exit(0)
    ds = ['T-drive', 'GeoLife', 'Roma', 'Porto', 'TaxiSH', 'MC', 'SFBA', 'AIS', 'ASL2']
    d = ds[0]
    d = 'AIS24-1-3'
    print(d)
    ts = load(d)
    if len(ts) < 4:
        ts = ts[0]
    if len(ts) < 4:
        ts = ts[0]
    print(len(ts))