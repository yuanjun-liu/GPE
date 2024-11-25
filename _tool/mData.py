import numpy as np
import sys
import _pickle
import zlib
import random
alpha = 'abcdefghijklmnopqrstuvwxyz'

def hammingDistance(x: int, y: int):
    return bin(x ^ y).count('1')

def onlyLetter(ss):
    return ''.join(filter(str.isalpha, ss)).lower()

def random_str(n):
    s = alpha + alpha.upper() + '1234567890'
    return ''.join(random.choices(s, k=n))
__dicted_idx = {}
__global_dict = {}

def dicted_idx_clear():
    __dicted_idx.clear()

def dicted_idx(type, x=None):
    """
    >>> dicted_idx('type','a')
    0
    >>> dicted_idx('type','b')
    1
    >>> dicted_idx('type')
    ['a','b']
    """
    if type not in __dicted_idx:
        __dicted_idx[type] = {}
    if x is None:
        return list(__dicted_idx[type].keys())
    if x in __dicted_idx[type]:
        return __dicted_idx[type][x]
    i = len(__dicted_idx[type])
    __dicted_idx[type][x] = i
    return i

class inc_idx:

    def __init__(self, start=0):
        self.i = start - 1

    def __call__(self, *args, **kwargs):
        self.i += 1
        return self.i

def global_dict(key, val=None):
    """global_dict('a',1); global_dict('a')->1"""
    global __global_dict
    assert key is not None
    if val is None:
        return __global_dict[key] if key in __global_dict else None
    __global_dict[key] = val

def file_dict(key, val=None):
    file = 'file_dict.pk'
    try:
        with open(file, 'rb') as f:
            d = _pickle.load(f)
    except:
        d = {}
    assert key is not None
    if val is None:
        return d[key] if key in __global_dict else None
    d[key] = val
    with open(file, 'wb') as f:
        _pickle.dump(d, f, protocol=5)
is_debug = True if sys.gettrace() else False

def isviadecorator():
    import inspect
    for fram in inspect.stack():
        if fram.code_context is not None:
            try:
                if fram.code_context[0].lstrip(' ')[0] == '@':
                    return True
            except Exception as e:
                print('error mData.isviadecorator', e)
    return False

def str2int_float_str(s):
    if isinstance(s, int) or isinstance(s, float):
        return s
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return s

def str2b(x, encoding='utf-8'):
    return bytes(x, encoding=encoding)

def b2str(x, encoding='utf-8'):
    return str(x, encoding=encoding)

def serialize(data):
    return _pickle.dumps(data)

def deserialize(data):
    return _pickle.loads(data)

def int2b(x, len=None):
    len = int(np.ceil(np.log(x) / np.log(255))) if len is None else len
    return int(x).to_bytes(length=len, byteorder='big', signed=True)

def b2int(x):
    return int().from_bytes(x, byteorder='big', signed=True)

def str_zip(s: str):
    return zlib.compress(str.encode(s), zlib.Z_BEST_COMPRESSION)

def str_unzip(x):
    return zlib.decompress(x).decode('utf-8')

def zhifang(x):
    a, b = (int(min(x)), int(max(x)))
    zf = np.zeros(b - a + 1)
    for i in x:
        zf[int(i - a)] += 1
    return zf

def hashable(x):
    try:
        hash(x)
        return True
    except:
        return False
if __name__ == '__main__':
    pass