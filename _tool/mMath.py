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

