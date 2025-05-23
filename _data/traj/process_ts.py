import random
import numpy as np
from typing import List
import sys, os
sys.path.extend(['./', '../../', '../'])
from _data.vector.process_point import ps_bbox

def ts_bbox(ts) -> List[List[float]]:
    """[min_lat,max_lat],[min_lon,max_lon],[min_tim,max_tim],..."""
    res = None
    i = 0
    while res is None:
        res = ps_bbox(ts[i])
        i += 1
    for t in ts[i:]:
        box = ps_bbox(t)
        if box is None:
            continue
        for i, b in enumerate(box):
            res[i][0] = min(res[i][0], b[0])
            res[i][1] = max(res[i][1], b[1])
    return res

def t_len(t: np.ndarray) -> float:
    return sum([np.linalg.norm(t[i, :2] - t[i + 1, :2]) for i in range(len(t) - 1)])

def ts_len_info(trajs) -> dict:
    len_min, len_max = (float('inf'), float('-inf'))
    len_sum, len_avg = (0, 0)
    num_traj = len(trajs)
    for T in trajs:
        l = len(T)
        len_min = min(l, len_min)
        len_max = max(l, len_max)
        len_sum += l
    len_avg = len_sum / num_traj
    return {'num': num_traj, 'len_max': len_max, 'len_min': len_min, 'len_avg': int(len_avg)}

def ts_split_2zip(ts, min_len=5):
    """ts -> 1,3,5...;2,4,6... -> A,B """
    A, B = ([], [])
    for t in ts:
        ta, tb = (t[::2, :], t[1::2, :])
        if len(ta) >= min_len and len(tb) >= min_len:
            A.append(ta)
            B.append(tb)
    return (A, B)

def ts_split_step(ts, max_len=50, min_len=15, step_len=10):
    """ts -> 1234;3456;... -> ts"""
    TS = []
    for t in ts:
        for i in range(0, len(t), step_len):
            j = min(len(t), i + max_len)
            if j - i < min_len:
                continue
            TS.append(t[i:j])
    return TS

def ts_bound(ts, xx_y_bound, tlen_min=1e-06, pnum_min=10):
    [[xmin, xmax], [ymin, ymax]] = xx_y_bound
    TS = []
    for t in ts:
        T = []
        for p in t:
            if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax:
                T.append(p)
        T = np.array(T)
        if len(T) >= pnum_min and t_len(T) > tlen_min:
            TS.append(T)
    return np.array(TS, dtype=object)

def ts_shrink(TS, del_rate):
    """:return ts,bbox"""
    assert 0 < del_rate < 1
    x = ts_bbox(TS)
    [xa, xb], [ya, yb] = (x[0], x[1])
    h = 10000
    dx, dy = ((xb - xa) / h, (yb - ya) / h)
    xns = np.zeros(h + 1, dtype=int)
    yns = np.zeros(h + 1, dtype=int)
    p2i = lambda p: [int((p[0] - xa) / dx), int((p[1] - ya) / dy)]
    for t in TS:
        for p in t:
            xi, yi = p2i(p)
            xns[xi] += 1
            yns[yi] += 1
    num = sum(xns)
    for i in range(h):
        if sum(xns[:i]) / num <= del_rate:
            x_min = xa + i * dx
        if sum(xns[i:]) / num >= del_rate:
            x_max = xa + i * dx
        if sum(yns[:i]) / num <= del_rate:
            y_min = ya + i * dy
        if sum(yns[i:]) / num >= del_rate:
            y_max = ya + i * dy
    bbox = [[x_min, x_max], [y_min, y_max]]
    return (ts_bound(TS, bbox), bbox)
