import heapq
import warnings
import numpy as np
import ctypes, sys
from typing import Any, Iterator, List, Tuple
from collections.abc import Iterable

def memory_list(a, types=[]) -> int:
    m = sys.getsizeof(a)
    if not iterable(a):
        return m
    for i in a:
        if isinstance(i, list) or isinstance(i, tuple):
            m += memory_list(i)
        elif isinstance(i, dict):
            for k in i:
                m += memory_list(k) if iterable(k) else sys.getsizeof(k)
                m += memory_list(i[k]) if iterable(i[k]) else sys.getsizeof(i[k])
        elif isinstance(a, dict):
            m += memory_list(i) if iterable(i) else sys.getsizeof(i)
            m += memory_list(a[i]) if iterable(a[i]) else sys.getsizeof(a[i])
        else:
            m += sys.getsizeof(i)
    return m

def iterable_pure(x):
    return isinstance(x, Iterable)

def iterable(x):
    return not isinstance(x, str) and isinstance(x, Iterable) and (len(x) > 0)

def deep_copy(x):
    if isinstance(x, np.ndarray):
        res = []
        for i in range(len(x)):
            res.append(deep_copy(x[i]))
        return np.array(res, dtype=x.dtype)
    elif isinstance(x, list):
        res = []
        for i in range(len(x)):
            res.append(deep_copy(x[i]))
        return res
    else:
        return x

def get_deep(x):
    return get_deep(x[0]) + 1 if iterable(x) else 0

def deep_yield(x, keep_dim=0):
    if get_deep(x) > keep_dim:
        for i in x:
            yield from deep_yield(i, keep_dim)
    else:
        yield x
def deep_filter(x, func, keep_dim):
    def _deep_filter(x, func, deep):
        if deep == keep_dim + 1:
            i = 0
            while i < len(x):
                if not func(x[i]):
                    del x[i]
                else:
                    i += 1
        else:
            for y in x:
                _deep_filter(y, func, deep - 1)
    deep = get_deep(x)
    _deep_filter(x, func, deep)
    return x

def deep_flatten(x, keep_dim=0):
    res = []
    for x in deep_yield(x, keep_dim):
        res.append(x)
    return res

def flatten(x: list):
    x = deep_copy(x)
    i = 0
    while i < len(x):
        if isinstance(x[i], list):
            v = x.pop(i)
            x.extend(v)
        else:
            i += 1
    return x

def flatten_times(x: list, t=1):
    x = deep_copy(x)
    for _ in range(t):
        x = flatten_fast(x)
    return x

def flatten_fast(x: list):
    return sum(x, [])

def list_reshape(li: list, shape: list):
    def val(id):
        return ctypes.cast(id, ctypes.py_object).value

    def stack(li, w):
        h = len(li) // w
        a, tail = (li[:h * w], li[h * w:])
        adr = np.array([id(x) for x in a]).reshape([h, w])
        if w > 1:
            a = [[val(int(adr[j, i])) for i in range(w)] for j in range(h)]
        else:
            a = [val(int(adr[j, 0])) for j in range(h)]
        if len(tail) > 1:
            a.append(tail)
        return a
    if len(shape) < 1:
        return li
    for s in shape[1:]:
        assert isinstance(s, int), 'shape is not int'
    for w in shape[::-1]:
        li = stack(li, w)
    return li

def _test_lsit_reshape():
    a = ['as', 1, [1, 1], {1: 2, 3: 4}, 3.5, (1, 2), None, object]
    b = list_reshape(a, [4])
    assert b == [['as', 1, [1, 1], {1: 2, 3: 4}], [3.5, (1, 2), None, object]]
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    b = list_reshape(a, [2, 3])
    assert b == [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11]]]
    print(list_reshape([1, 2, 3, 4, 5, 6, 7, 8], [2, 2]))
    print(list_reshape([1, 2, 3, 4, 5], [3]))

def zipxs(l1, l2, *ls):
    if len(l1) == 0 or len(l2) == 0:
        return []
    if isinstance(l1[0], list):
        l = [[*i, j] for i in l1 for j in l2]
    else:
        l = [[i, j] for i in l1 for j in l2]
    if len(ls) == 0:
        return l
    return zipxs(l, ls[0], *ls[1:])

def T(x):
    return zip(*x)

def partition(x, parts):
    assert sum(parts) <= 1
    assert all([p >= 0 for p in parts])
    res = []
    last = 0
    for p in parts:
        l = int(len(x) * p)
        res.append(x[last:last + l])
        last += l
    return res

def reverse2(x: list):
    H, W = (len(x), max([len(i) for i in x]))
    res = [[] for i in range(W)]
    for h in range(H):
        for w in range(len(x[h])):
            res[w].append(x[h][w])
    return res

def shuffle(data, *others):
    data = [data, *others]
    for d in data:
        assert iterable(d)
        assert len(d) == len(data[0])
    idx = np.array([i for i in range(len(data[0]))])
    np.random.shuffle(idx)
    for dat in data:
        for i in range(len(idx)):
            dat[i], dat[idx[i]] = (dat[idx[i]], dat[i])
    return data if len(others) > 0 else data[0]

def batch_iter(bs, data, *others):
    data = [data, *others]
    for d in data:
        assert iterable(d)
        assert len(d) == len(data[0])
    for i in range(len(data[0]) // bs):
        batch = [dat[i * bs:min(len(data[0]), (i + 1) * bs)] for dat in data]
        yield (batch if len(others) > 0 else batch[0])

def topki(a, k):
    assert k <= len(a)
    assert isinstance(a, list) or isinstance(a, np.ndarray)
    islist = isinstance(a, list)
    b = np.array(a) if islist else a
    assert len(b) > 0
    xi = np.argwhere(b >= min(heapq.nlargest(k, b))).reshape(-1)
    if islist:
        return list(xi)
    return xi

def topk(a, k):
    assert k <= len(a)
    if isinstance(a, list):
        return heapq.nlargest(k, a, a.__getitem__)
    if isinstance(a, np.ndarray):
        return heapq.nlargest(k, a)
    raise 'unsupport dtype'

def bisearch_mstep(sorted: list, x, mstep: int) -> int:
    """只走m步的近似二分搜索"""
    l, r = (0, len(sorted) - 1)
    for _ in range(mstep):
        if l >= r:
            return l
        h = (l + r) // 2
        if sorted[h] < x:
            l = h + 1
        elif sorted[h] > x:
            r = h
        else:
            return h
    return l

class ChoiceIdxs:
    """m步近似搜索，构造O(n)，搜索O(m) or log(n)"""

    def __init__(self, p: list) -> None:
        self.p = [x for x in p]
        assert self.p[0] >= 0
        for i in range(1, len(p)):
            assert self.p[i] >= 0
            self.p[i] += self.p[i - 1]
        for i in range(len(p)):
            self.p[i] /= self.p[-1]

    def __call__(self, mstep=None) -> Any:
        mstep = int(np.log(len(self.p))) + 1 if mstep is None else mstep
        return bisearch_mstep(self.p, np.random.random(), mstep)

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

def append_line(cols: list, data):
    assert len(cols) == len(data)
    for i, d in enumerate(data):
        cols[i].append(d)
    return cols

def for_mutiable(a: list):
    """不适用于包含多个相同元素"""
    ids = []
    while True:
        full = True
        for x in a:
            idx = id(x)
            if idx not in ids:
                ids.append(idx)
                full = False
                yield x
                break
        if full:
            break

class MutiableFor(Iterable):
    def __init__(self, data: list) -> None:
        self.data = data
        self.ids = []

    def __iter__(self):
        return self

    def __next__(self):
        for x in self.data:
            idx = id(x)
            if idx not in self.ids:
                self.ids.append(idx)
                return x
        raise StopIteration()

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
if __name__ == '__main__':
    data = np.zeros((2, 3, 4, 5))
    pass