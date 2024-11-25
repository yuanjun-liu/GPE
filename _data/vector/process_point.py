from typing import List
import numpy as np

def ps_bbox(ps: np.ndarray) -> List[List[float]]:
    """[min_lat,max_lat],[min_lon,max_lon],[min_tim,max_tim],..."""
    if len(ps) == 0:
        return None
    ps = np.array(ps)
    return [[ps[:, x].min(), ps[:, x].max()] for x in range(len(ps[0]))]

def p_in_box(p, bbox):
    ok = True
    for i, b in enumerate(bbox):
        if not b[0] <= p[i] <= b[1]:
            ok = False
            break
    return ok

def ps_bound(ps: np.ndarray, bbox) -> np.ndarray:
    return np.array([p for p in ps if p_in_box(p, bbox)])