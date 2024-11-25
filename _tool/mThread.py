import sys
sys.path.extend(['../', './'])
import time
import multiprocessing
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process
import threading, os
import subprocess
from functools import wraps
from subprocess import PIPE, Popen
from _tool.mData import str2b, b2str
from _tool.mList import zipxs, iterable, deep_flatten
from _tool.mFile import check_dir
from multiprocessing.dummy import Lock as ThreadLock
from multiprocessing import Lock as ProcessLock

def wait_break(fun, arg, timeout):
    p = ThreadPool(1)
    if iterable(arg):
        if len(arg) > 1:
            res = p.apply_async(fun, args=arg)
        else:
            res = p.apply_async(fun, args=(arg[0],))
    else:
        res = p.apply_async(fun, args=(arg,))
    try:
        out = res.get(timeout)
        return out
    except multiprocessing.TimeoutError:
        print('Aborting ' + fun.__name__ + ' due to timeout')
        return None

def wait_break_dec(time):
    """do not support kw"""

    def decorator(func):

        @wraps(func)
        def wrapper(*args):
            return wait_break(func, args, time)
        return wrapper
    return decorator
_no_support = 'File Locker only support NT and Posix platforms!'
if os.name == 'nt':
    import win32con, win32file, pywintypes
    LOCK_EX = win32con.LOCKFILE_EXCLUSIVE_LOCK
    LOCK_SH = 0
    LOCK_NB = win32con.LOCKFILE_FAIL_IMMEDIATELY
    _overlapped = pywintypes.OVERLAPPED()
elif os.name == 'posix':
    import fcntl
    from fcntl import LOCK_EX, LOCK_SH, LOCK_NB
else:
    raise RuntimeError(_no_support)

def run_cmd0(cmd):
    return os.system(cmd)

def run_cmd1(cmd, *arg):
    mode = 'r' if len(arg) == 0 else 'w'
    p = os.popen(cmd, mode, 1024)
    if mode == 'w':
        for a in arg:
            p.write(str(a) + ' ')
        res = []
    else:
        res = p.readlines()
        res = [r.replace('\n', '') for r in res]
    p.close()
    return (res, p.errors)

def run_cmd2(cmd, *input):
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=False)
    i = str2b(' '.join([str(i) for i in input]))
    out, err = proc.communicate(i)
    proc.kill()
    try:
        out, err = (b2str(out), b2str(err))
    except:
        out, err = (b2str(out, 'gbk'), b2str(err, 'gbk'))
    out = out.replace('\r', '').split('\n')
    err = err.replace('\r', '').split('\n')
    return (out, err)

class FileLock:

    def __init__(self, f=None, flag=LOCK_EX):
        _f = './logs/file.lock'
        check_dir(_f)
        if f is None:
            self.__f = open(_f, 'a')
        elif isinstance(f, str):
            self.__f = open(f, 'a')
        else:
            self.__f = f
        self.__locked = False
        self.__flag = flag

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.release()

    def acquire(self):
        if os.name == 'nt':
            hfile = win32file._get_osfhandle(self.__f.fileno())
            win32file.LockFileEx(hfile, self.__flag, 0, 4294901760, _overlapped)
        elif os.name == 'posix':
            fcntl.flock(self.__f.fileno(), self.__flag)
        else:
            raise RuntimeError(_no_support)
        self.__locked = True
        return self

    def release(self):
        if os.name == 'nt':
            hfile = win32file._get_osfhandle(self.__f.fileno())
            win32file.UnlockFileEx(hfile, 0, 4294901760, _overlapped)
        elif os.name == 'posix':
            fcntl.flock(self.__f.fileno(), fcntl.LOCK_UN)
        else:
            raise RuntimeError(_no_support)
        self.__locked = False
        return self

    def locked(self):
        return self.__locked
plock_default = ProcessLock()
tlock_default = ThreadLock()
flock_default = FileLock()

def fun_fenfa(arg):
    return arg[0](*arg[1])

def execPool(func, args, pool_size=8, keep_dim=False, dummy=False, asfor=False):
    if args is None or not iterable(args) or len(args) == 0:
        print('ERROR at execPool, arg is empty')
        return
    if pool_size > 1 and (not asfor):
        if dummy:
            pool = ThreadPool(pool_size)
        else:
            pool = ProcessPool(pool_size)
    else:
        asfor = True
    if not keep_dim and len(args) == 1 or not iterable(args[0]) or (iterable(args[0]) and len(args[0]) == 1):
        if iterable(args[0]):
            args = deep_flatten(args, 0)
        if asfor:
            return [func(a) for a in args]
        res = pool.map(func, args)
    else:
        if asfor:
            return [func(*a) for a in args]
        arg = zipxs([func], args)
        res = pool.map(fun_fenfa, arg)
    pool.close()
    pool.join()
    return res

def exePool_simple(func, args=[], pool_size=8, thread=False):
    if thread:
        pool = ThreadPool(pool_size)
    else:
        pool = ProcessPool(pool_size)
    res = pool.map(fun_fenfa, [(func, a) for a in args])
    pool.close()
    pool.join()
    return res

def newThread(func, *arg, **kw):
    t = threading.Thread(target=func, args=arg, kwargs=kw)
    t.start()
    return t

def newProcess(func, *arg, **kw):
    p = Process(target=func, args=arg, kwargs=kw)
    p.start()
    return p

def __tobe_invoke(tim, fun, *arg):
    time.sleep(tim)
    return fun(*arg)

def invokeProcess(tim, fun, *arg):
    return newProcess(__tobe_invoke, tim, fun, *arg)

def invokeThread(tim, fun, *arg):
    return newThread(__tobe_invoke, tim, fun, *arg)

def invoke(tim, fun, *arg):
    return __tobe_invoke(tim, fun, *arg)

def wait_for(fun_isok, fun1_arg=None, fun_ok=None, fun2_arg=None, fun_bad=None, fun3_arg=None, max_wait_time=30, dt=0.1):
    assert fun_isok is not None
    t = 0
    while t < max_wait_time:
        if fun1_arg is not None:
            res = fun_isok(*fun1_arg)
        else:
            res = fun_isok()
        if res:
            if fun_ok is not None:
                if fun2_arg is not None:
                    fun_ok(*fun2_arg)
                else:
                    fun_ok()
            return True
        t += dt
        time.sleep(dt)
    if fun_bad is not None:
        if fun3_arg is not None:
            fun_bad(*fun3_arg)
        else:
            fun_bad()
    return False

def __pool_test(a, b=0):
    import time
    time.sleep(a)
    print(a + b)
    return a + b

def __file_test(a):
    import time
    time.sleep(a)
    print(a)
    with open('t', 'a', encoding='utf-8') as f:
        print(a)
        with FileLock(f) as lock:
            print(lock.locked())
            f.write(str(a))

def __wait_break_test():

    @wait_break_dec(10)
    def f2(x):
        time.sleep(x)
        return x
    print(f2(1))
if __name__ == '__main__':
    pass
    __wait_break_test()