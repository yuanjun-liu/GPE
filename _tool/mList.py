import heapq
import warnings
import numpy as np
import ctypes, sys
from typing import Any, Iterator, List, Tuple
from collections.abc import Iterable


def iterable(x):
    return not isinstance(x, str) and isinstance(x, Iterable) and (len(x) > 0)


def flatten_fast(x: list):
    return sum(x, [])


def T(x):
    return zip(*x)


def choice(data: List or np.ndarray, num: int, replace=False, p=None):
    assert isinstance(data, list) or isinstance(data, tuple) or isinstance(data, np.ndarray)
    if num > len(data) and replace == False:
        replace = True
        warnings.warn('in choice, num>len(data), we set replace=True')
    mask = np.random.choice(len(data), num, replace=replace, p=p)
    is_list = isinstance(data, list) or isinstance(data, tuple)
    dat = np.array(data, dtype=object) if is_list else data.copy()
    dat = dat[mask]
    return list(dat) if is_list else dat

def choice_co(dataS: List | np.ndarray, num: int, replace=False, p=None):
    le = len(dataS[0])
    assert num <= le
    mask = np.random.choice(le, num, replace=replace, p=p)
    res = []
    for data in dataS:
        assert isinstance(data, list) or isinstance(data, np.ndarray)
        assert len(data) == le
        is_list = isinstance(data, list)
        dat = np.array(data, dtype=object) if is_list else data.copy()
        dat = dat[mask]
        res.append(dat.tolist() if is_list else dat)
    return res


def mdidx(dat: np.ndarray, idx_all, keep_dims=False):
    assert len(dat.shape) == len(idx_all)
    for i in range(len(idx_all)):
        if isinstance(idx_all[i], list):
            assert max(idx_all[i]) < dat.shape[i]
        else:
            assert idx_all[i] < dat.shape[i]

    def slice(dat: np.ndarray, idx_all) -> np.ndarray:
        if len(idx_all) == 1:
            return dat[idx_all[0]]
        if isinstance(idx_all[0], list):
            return np.stack([slice(dat[i], idx_all[1:]) for i in idx_all[0]])
        else:
            return slice(dat[idx_all[0]], idx_all[1:])

    def s2_keep(dat: np.ndarray, idx_all) -> np.ndarray:
        if len(idx_all) == 1:
            return dat[idx_all[0]].reshape(-1)
        if isinstance(idx_all[0], list):
            return np.stack([s2_keep(dat[i], idx_all[1:]) for i in idx_all[0]])
        else:
            return np.expand_dims(s2_keep(dat[idx_all[0]], idx_all[1:]), 0)
    return s2_keep(dat, idx_all) if keep_dims else slice(dat, idx_all)
