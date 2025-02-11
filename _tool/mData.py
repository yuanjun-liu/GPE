import numpy as np
import sys
import _pickle
import zlib
import random


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


