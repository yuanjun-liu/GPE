from typing import List
import numpy as np

def ps_bbox(ps: np.ndarray) -> List[List[float]]:
    """[min_lat,max_lat],[min_lon,max_lon],[min_tim,max_tim],..."""
    if len(ps) == 0:
        return None
    ps = np.array(ps)
    return [[ps[:, x].min(), ps[:, x].max()] for x in range(len(ps[0]))]

