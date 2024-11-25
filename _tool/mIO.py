from functools import wraps
import sys, os, _pickle
sys.path.extend(['./'])
from _tool.mData import isviadecorator, str2int_float_str
from _tool.mList import iterable
from _tool.mFile import cache_dir, check_file, check_dir, out_dir
import json

def save(file, stuff):
    dir = os.path.dirname(file)
    if not os.path.exists(dir):
        os.makedirs(dir)
    if file.endswith('.npy'):
        return save_np(file, stuff)
    if file.endswith('.txt'):
        return save_txt(file, stuff)
    if file.endswith('.pk'):
        return save_pk(file, stuff)
    if file.endswith('.th'):
        return save_th(file, stuff)
    if file.endswith('.csv'):
        return save_csv(file, stuff)
    if file.endswith('.json'):
        return save_json(file, stuff)
    if file.endswith('.yml') or file.endswith('.yaml'):
        return save_yaml(file, stuff)
    raise TypeError('不支持的格式')

def load(file):
    if file.endswith('.npy'):
        return load_np(file)
    if file.endswith('.txt'):
        return load_txt(file)
    if file.endswith('.pk'):
        return load_pk(file)
    if file.endswith('.th'):
        return load_th(file)
    if file.endswith('.csv'):
        return load_csv(file)
    if file.endswith('.json'):
        return load_json(file)
    if file.endswith('.yml') or file.endswith('.yaml'):
        return load_yaml(file)
    raise TypeError('不支持的格式')

def load_json(file):
    with open(file, 'r') as f:
        return json.load(f)

def save_json(file, dic: dict):
    assert isinstance(dic, dict)
    with open(file, 'w') as f:
        json.dump(dic, f)

def load_yaml(file):
    import yaml
    with open(file, encoding='utf-8') as file1:
        data = yaml.load(file1, Loader=yaml.FullLoader)
    return data

def save_yaml(file, stuff):
    import yaml
    with open(file, 'w', encoding='utf-8') as f:
        yaml.dump_all(documents=stuff, stream=f, allow_unicode=True)

def save_csv(file, data):
    with open(file, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(','.join(d) + '\n')

def load_csv(file):
    raw = load_txt(file)
    m = [l.rstrip(',').split(',') for l in raw]
    for x in range(len(m)):
        for y in range(len(m[x])):
            m[x][y] = str2int_float_str(m[x][y].strip())
    return m

def _tostr(x):
    return '\t'.join(_tostr(x)) if iterable(x) else x

def save_txt(file, lines):
    with open(file, 'w', encoding='utf-8') as f:
        for line in lines:
            if not isinstance(line, str):
                line = _tostr(line)
            f.write(line + '\n')

def load_txt(file):
    with open(file, 'r', encoding='utf-8') as f:
        d = f.read()
        if d.endswith('\n'):
            d = d.rstrip('\n')
        return d.split('\n')

def save_pk(file, stuff):
    with open(file, 'wb') as f:
        _pickle.dump(stuff, f, protocol=5)

def load_pk(file):
    assert os.path.exists(file), ' no such file:' + file
    with open(file, 'rb') as f:
        return _pickle.load(f)

def save_th(file, stuff):
    from torch import save as thsave
    thsave(stuff, file)

def load_th(file):
    assert os.path.exists(file), ' no such file:' + file
    import torch
    from torch import load as thload
    try:
        return thload(file)
    except RuntimeError as e:
        return thload(file, map_location=torch.device('cpu'))

def save_np(file, stuff):
    import numpy as np
    np.save(file, stuff)

def load_np(file):
    import numpy as np
    if not file.endswith('npy'):
        file += '.npy'
    assert os.path.exists(file), ' no such file:' + file
    return np.load(file, allow_pickle=True)

def grip_fun_kwarg(func, key, kwargs, kw2=None):
    val = None
    if key in kwargs:
        val = kwargs[key]
        if func is not None:
            if key not in func.__code__.co_varnames:
                kwargs.pop(key)
    if val is None and kw2 is not None:
        return grip_fun_kwarg(func, key, kw2)
    return val

def name_fun_arg(name, *arg, **kwarg):
    res = []
    for i, val in enumerate(arg):
        if iterable(val):
            continue
        if isinstance(val, list):
            continue
        if isinstance(val, dict):
            continue
        if isinstance(val, set):
            continue
        res.append(f'{val}')
    for key in kwarg:
        val = kwarg[key]
        if iterable(val):
            continue
        res.append(f'{val}')
    return name + '(' + '_'.join(res) + ')'

def mcache(name=None, *data, **kws):
    def cache_decorator(func):
        redir = grip_fun_kwarg(func, 'redir', kws)
        dir = grip_fun_kwarg(func, 'dir', kws)
        debug = grip_fun_kwarg(func, 'debug', kws)
        ftype = grip_fun_kwarg(func, 'ftype', kws)
        if ftype is None:
            ftype = 'pk'
        file_base = out_dir(dir) if dir else cache_dir()
        if redir is not None:
            file_base = redir

        @wraps(func)
        def wrapped_mIO(*args, **kwargs):
            return_if_exist = grip_fun_kwarg(func, 'return_if_exist', kwargs)
            name2 = name_fun_arg(func.__name__ if name is None else name, *args, **kwargs)
            file = os.path.join(file_base, name2) + '.' + ftype
            file_exist = check_file(file)
            if debug:
                print(file, ' ' if file_exist else ' not', 'exist')
            if file_exist:
                if debug:
                    print('load cache:', file)
                if return_if_exist:
                    return True
                try:
                    data = load(file)
                    return data
                except EOFError as e:
                    print('load fail, recaculate:', name2)
                    pass
            data = func(*args, **kwargs)
            if debug:
                print('save cache:', file)
            check_dir(file)
            save(file, data)
            return data
        return wrapped_mIO
    if isviadecorator():
        return cache_decorator
    else:
        assert name is not None, 'cache need name'
        redir = grip_fun_kwarg(None, 'redir', kws)
        ftype = grip_fun_kwarg(None, 'ftype', kws)
        test_exist = grip_fun_kwarg(None, 'test_exist', kws)
        debug = grip_fun_kwarg(None, 'debug', kws)
        if ftype is None:
            ftype = 'pk'
        file_base = cache_dir()
        if redir is not None:
            file_base = redir
        file = os.path.join(file_base, name) + '.' + ftype
        check_dir(file)
        file_exist = check_file(file)
        if test_exist:
            return file_exist
        if debug:
            print(file, '' if file_exist else 'not', ' exist')
        if data is None or len(data) == 0:
            if not file_exist:
                return None
            if debug:
                print('load cache:', file)
            data = load(file)
            return data
        if debug:
            print('save cache:', file)
        save(file, data)

def cache_ifno_build(name, build_fun, build_arg=None, mcache_kw=None):
    res = mcache(name) if mcache_kw is None else mcache(name, **mcache_kw)
    if res is not None:
        return res
    res = build_fun() if build_arg is None else build_fun(*build_arg)
    mcache(name, res) if mcache_kw is None else mcache(name, res, **mcache_kw)
    return res

def set_env(name: str, value: str):
    name = 'py_env_' + name
    os.environ[name] = value

def get_env(name: str):
    name = 'py_env_' + name
    if name not in os.environ:
        return None
    return os.environ[name]
if __name__ == '__main__':
    pass