import sys
sys.path.extend(['../', './'])
import time
import multiprocessing
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Process
import threading, os
import subprocess
from _tool.mFile import check_dir
from multiprocessing.dummy import Lock as ThreadLock
from multiprocessing import Lock as ProcessLock


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

