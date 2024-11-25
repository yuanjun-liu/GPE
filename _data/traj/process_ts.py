import random
import numpy as np
from typing import List
import sys, os
sys.path.extend(['./', '../../', '../'])
from _data.vector.process_point import ps_bbox
from _tool.mList import deep_filter

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

def t2ps_its(T: np.ndarray, h: int):
    """list of p:x,y,t"""
    l = t_len(T)
    vec = np.array([T[0]] * h)
    vec[-1] = T[-1]
    if l < 1e-12:
        return vec
    e = l / (h - 1)
    a, b = (0, 0)
    j = 0
    for i in range(1, len(T)):
        p1, p2 = (T[i - 1], T[i])
        s = np.linalg.norm(p1[:2] - p2[:2], 2)
        if s < 1e-12:
            continue
        b += s
        while b >= a:
            if j >= h:
                break
            vec[j] = (p2 - p1) * (1 - (b - a) / s) + p1
            j += 1
            a += e
    return vec

def its(T1, T2, h):
    return np.linalg.norm(t2ps_its(T1, h) - t2ps_its(T2, h)) / np.sqrt(h)

def t2ps_time(T: np.ndarray, h: int):
    """list of p:x,y,t"""
    l = T[-1][2] - T[0][2]
    vec = np.array([T[0]] * h)
    vec[-1] = T[-1]
    if abs(l) < 1e-12:
        return vec
    e = l / (h - 1)
    a, b = (0, 0)
    j = 0
    for i in range(1, len(T)):
        p1, p2 = (T[i - 1], T[i])
        s = p2[2] - p1[2]
        if s < 1e-12:
            continue
        b += s
        while b >= a:
            if j >= h:
                break
            vec[j] = (p2 - p1) * (1 - (b - a) / s) + p1
            j += 1
            a += e
    return vec

def t2ps_steplen(T: np.ndarray, e: float):
    """list of p:x,y,t"""
    l = t_len(T)
    vec = []
    if l < e:
        return np.array([T[0]])
    a, b = (0, 0)
    j = 0
    for i in range(1, len(T)):
        p1, p2 = (T[i - 1], T[i])
        s = np.linalg.norm(p1[:2] - p2[:2])
        if s < 1e-12:
            continue
        b += s
        while b >= a:
            vec.append((p2 - p1) * (1 - (b - a) / s) + p1)
            j += 1
            a += e
    vec.append(T[-1])
    return np.array(vec)

def t2ps_nocover(T: np.ndarray, e: float):
    """list of p:x,y,t"""
    ee = e / 10
    dis = lambda x, y: np.linalg.norm(x[:2] - y[:2])
    ps = [T[0]]
    vec = np.array(ps)
    a, b = (0, 0)
    for i in range(1, len(T)):
        p1, p2 = (T[i - 1], T[i])
        s = dis(p1, p2)
        if s < 1e-12:
            continue
        b += s
        while b >= a:
            cur = (p2 - p1) * (1 - (b - a) / s) + p1
            if all(np.sqrt(np.sum(np.power(vec - cur, 2), axis=1)) >= e):
                ps.append(cur)
                vec = np.array(ps)
            a += ee
    if all(np.sqrt(np.sum(np.power(vec - T[-1], 2), axis=1)) >= e):
        ps.append(T[-1])
    return np.array(ps)

def t2ps_nocover_simple(T: np.ndarray, e: float):
    """list of p:x,y,t"""
    ee = e / 3
    dis = lambda x, y: np.linalg.norm(x[:2] - y[:2])
    ps = [T[0]]
    a, b, c = (0, 0, 0)
    for i in range(1, len(T)):
        p1, p2 = (T[i - 1], T[i])
        s = dis(p1, p2)
        if s < 1e-12:
            continue
        b += s
        while b >= a:
            if c >= e:
                cur = (p2 - p1) * (1 - (b - a) / s) + p1
                if dis(cur, ps[-1]) >= e:
                    ps.append(cur)
                    c = 0
            a += ee
            c += ee
    return np.array(ps)

def t2gs(T: np.ndarray, xa, xb, ya, yb, xh, yh):
    """:return [g:[gxi,gyi,tim], p1, p2]"""
    res = []
    dx, dy = ((xb - xa) / xh, (yb - ya) / yh)
    ps = t2ps_steplen(T, min(dx, dy) / 3.0)
    p2g = lambda p: (int((p[0] - xa) / dx), int((p[1] - ya) / dy), p[2])
    if len(ps.shape) == 1:
        return [[p2g(ps), ps, ps]]
    g, p = (p2g(ps[0]), ps[0])
    for i in range(1, len(ps)):
        g2 = p2g(ps[i])
        if g2 != g:
            res.append([g, p, ps[i]])
            g, p = (g2, ps[i])
    return res

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

def ts_filter_len(trajs, len_min=30):
    lenok = lambda T: len(T) >= len_min
    trajs = deep_filter(trajs, lenok, 2)
    return trajs

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

def t_drop(T: np.ndarray, rate):
    res = []
    for p in T:
        if random.random() > rate:
            res.append(p)
    return np.ndarray(res)

def t_noise(T: np.ndarray, rate, size):
    res = []
    for p in T:
        if random.random() < rate:
            q = p.copy()
            q[:2] += np.random.random(2) * size
            res.append(q)
    return np.ndarray(res)

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

def MtxIts(ts):
    if len(ts) < 4:
        ts = ts[0]
    if len(ts) < 4:
        ts = ts[0]
    if len(ts) < 4:
        ts = ts[0]
    n = len(ts)
    h = int(sum(map(len, ts)) / n)
    tps = [t2ps_its(t, h) for t in ts]
    mtx = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            mtx[i, j] = np.linalg.norm(tps[i] - tps[j])
            mtx[j, i] = np.linalg.norm(tps[i] - tps[j])
    mtx /= np.sqrt(h)
    return mtx

def reverse(T):
    """[(x,y,t),..] 反转 根据间隔调整时间"""
    dt = [T[i + 1][2] - T[i][2] for i in range(len(T) - 1)]
    dt = dt[::-1]
    t0 = T[0][2]
    T = T[::-1]
    T[0][2] = t0
    for i in range(1, len(T)):
        T[i][2] = T[i - 1][2] + dt[i - 1]
    return T