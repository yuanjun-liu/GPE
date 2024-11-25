import sys
import numpy as np
from typing import Tuple, List
from collections.abc import Iterable
dis_eucid = lambda x, y: np.linalg.norm(x - y)

def multi(ns: List) -> float:
    x = 1
    for n in ns:
        x *= n
    return x

def to_base(i: List[int] or int, base_x: int or List[int], base_i=10) -> list:
    if isinstance(i, list):
        i, x = (0, i)
        for xi in x:
            i = i * base_i + xi
    if isinstance(base_x, list):
        assert i < multi(base_x)
        res = [0 for _ in base_x]
        for bi in range(len(base_x)):
            res[bi] = i % base_x[-bi - 1]
            i = i // base_x[-bi - 1]
        return res[::-1]
    else:
        res = []
        while True:
            res.append(i % base_x)
            i = i // base_x
            if i == 0:
                return res[::-1]

def to_alpha(i, alpha='abcdefghijklmnopqrstuvwxyz', spt=''):
    idx = to_base(i, len(alpha))
    return spt.join([alpha[i] for i in idx])

def sys_max_min():
    return (sys.maxsize, -sys.maxsize)

def num_max():
    return sys.maxsize

def arctan_extend(x, y):
    single = isinstance(x, int) or isinstance(x, float)
    if single:
        x, y = ([x], [y])
    if isinstance(x, list):
        x, y = (np.array(x), np.array(y))
    sig = x < 0
    x[sig] *= -1
    y[sig] *= -1
    a = np.arctan(y / x)
    a[np.isnan(a)] = 0
    a[sig] += np.pi
    if single:
        return a[0]
    return a

def asin_acos_group_np(sin: np.ndarray, cos: np.ndarray):
    asin, acos = (np.arcsin(sin), np.arccos(cos))
    sin_le0 = sin < 0
    cos_le0 = cos < 0
    asin[cos_le0] = np.pi - asin[cos_le0]
    asin = np.fmod(asin + np.pi, np.pi * 2) - np.pi
    acos[sin_le0] = -acos[sin_le0]
    return (asin, acos)

def xy2po(x, y):
    return (np.sqrt(x ** 2 + y ** 2), arctan_extend(x, y))

def po2xy(p, a):
    return (p * np.cos(a), p * np.sin(a))

def random_range(low, high, size=1):
    if isinstance(low, int) and isinstance(high, int):
        return np.random.randint(low, high, size, dtype=int)
    return np.random.random(size) * (high - low) + low

def safe_divide(a, b):
    if abs(b) < 1e-08:
        return num_max()
    return a / b

def int2b_list(a, fix_len=None):
    x = [int(x) for x in bin(a)[2:]]
    if fix_len is None:
        return x
    pre = [0] * (fix_len - len(x))
    pre.extend(x)
    return pre

def mean(*arg):
    if len(arg) == 1:
        arg = arg[0]
    return sum(arg) / len(arg)

def ranki_fast(x: np.ndarray):
    """1-D vec , 值越小 rank越小"""
    return np.argsort(x) + 1

def ranki_same(x: np.ndarray):
    idx = np.zeros_like(x, dtype=int)
    s = set(x)
    i = 1
    while s:
        mi = min(s)
        mask = x == mi
        idx[mask] = i
        i += sum(mask)
        s.remove(mi)
    return idx

def dis_p2l(p, lp1, lp2):
    if abs(dis_eucid(lp1, lp2)) < 0.001:
        return dis_eucid(p, lp1)
    x1, y1 = lp1
    x2, y2 = lp2
    x, y = p
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    fm = np.sqrt(np.power(A, 2) + np.power(B, 2))
    if abs(fm) < 0.0001:
        return dis_eucid(p, lp1)
    return abs(A * x + B * y + C) / fm

def dis_p2l3(p, lp1, lp2):
    a, b, c = (dis_eucid(p, lp1), dis_eucid(p, lp2), dis_eucid(lp1, lp2))
    p = (a + b + c) / 2
    s = np.sqrt(p * (p - a) * (p - b) * (p - c))
    return 2 * s / c

def dis_p2l2(q: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float, bool, bool]:
    q -= p1
    p2 -= p1
    p1 -= p1
    a = -arctan_extend(float(p2[0]), float(p2[1]))
    mtx = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    _p2 = np.matmul(mtx, p2)
    _q = np.matmul(mtx, q)
    return (abs(_q[1]), _q[0], _q[1] > 0, 0 <= _q[0] <= p2[0])

def intg(fun, xs_dx: Iterable):
    sum = 0
    for x, dx in xs_dx:
        sum += fun(x) * dx
    return sum

def dis_mtx_np(X, Y):
    x2 = np.sum(X ** 2, axis=1)
    y2 = np.sum(Y ** 2, axis=1)
    xy = np.matmul(X, Y.T)
    x2 = x2.reshape(-1, 1)
    return np.sqrt(x2 - 2 * xy + y2)

def dis_mtx_f(X, Y, f):
    n, m = (len(X), len(Y))
    dis = np.zeros((n, m))
    for i in range(n):
        for j in range(i):
            dis[j, i] = dis[i, j] = f(X[i], Y[j])
    return dis
if __name__ == '__main__':
    pass