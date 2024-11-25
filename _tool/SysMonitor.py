import json, time, os, sys
from typing import Any
from _tool.mThread import run_cmd1, ThreadLock, ProcessLock
import numpy as np
from _tool.mTime import now_date_time
import psutil
from _tool.mFile import check_dir, log_dir
from _tool.mList import T, append_line, flatten_fast, mdidx
from _tool.mIO import save, load
from _tool.mMath import multi, to_base, ranki_same

def mem_process(pid=None):
    """当前/指定pid 进程的内存占用"""
    pid = os.getpid() if pid is None else pid
    process = psutil.Process(pid)
    memInfo = process.memory_info()
    return memInfo.rss

def cpurate():
    return psutil.cpu_percent()

def memrate():
    return psutil.virtual_memory().percent

def memfree():
    return psutil.virtual_memory().free / 1024 / 1024

def CpuMemCheck(cpu_rate=None, mem_rate=None, mem_mb=None, timout=None, timout_callback=sys.exit):
    """
    block if current_cpu_rate > cpu_rate or current_mem_rate > mem_rate or current_mem_free < mem_mb
    :param cpu_rate: 0-100, None is 100
    :param mem_rate: 0-100, None is 100
    :param mem_mb: size MB of following code need, None is infinity
    :param timout: exit if wait more time that timout
    :param timout_callback: run timout_callback() if timout
    :return:
    """
    t = 0
    while True:
        ok = True
        if cpu_rate is not None and cpurate() > cpu_rate:
            ok = False
        if mem_rate is not None and memrate() > mem_rate:
            ok = False
        if mem_mb is not None and memfree() < mem_mb:
            ok = False
        if ok:
            break
        time.sleep(1)
        if timout is not None:
            t += 1
            if t > timout and timout_callback is not None:
                print('CpuMemCheck timout')
                timout_callback()

def profile(fun, arg, sort_cum_tot: bool, to_file: bool, print_num: float):
    funame = fun.__name__
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    print('profile', funame, ', ', *arg)
    timer = Timer()
    res = None
    with timer:
        res = fun(*arg)
    print('time:', timer.time_avg())
    print('result:\n', res, '\n')
    pr.disable()
    s = io.StringIO()
    sortby = 'cumtime' if sort_cum_tot else 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby).print_stats(print_num)
    ps.print_stats()
    print(s.getvalue())
    if to_file:
        pr.dump_stats(funame + '.prof')
        print('prof file at: {}.prof, and {}.csv'.format(funame, funame))
        run_cmd1('flameprof {}.prof > {}.svg'.format(funame, funame))

class Counter:

    def __init__(self, start=0, _flock=None):
        self._plock = ProcessLock()
        self._tlock = ThreadLock()
        self._flock = _flock
        with self._tlock:
            with self._plock:
                self.val = start - 1

    def __call__(self, *args: Any, **kwds: Any) -> int:
        with self._tlock:
            with self._plock:
                self.val += 1
                return self.val

class Timer:

    def __init__(self, cpu=False):
        self.time_times: int = 0
        self.time_sum: float = 0
        self.time_last: float = 0
        self.__time__doing: bool = False
        self.__time_cpu = cpu

    def tic(self, times=1):
        assert self.__time__doing == False, 'Timer tictoc error'
        self.__time__doing = True
        self.time_times += times
        self.time_last = time.process_time() if self.__time_cpu else time.time()

    def toc(self):
        t = time.process_time() if self.__time_cpu else time.time()
        assert self.__time__doing == True, 'Timer tictoc error'
        self.__time__doing = False
        self.time_sum += t - self.time_last

    def time_avg(self) -> float:
        return self.time_sum / self.time_times

    def __enter__(self):
        self.tic()

    def __exit__(self, *arg, **kw):
        self.toc()

    def time_clear(self):
        self.time_times = 0
        self.time_sum = 0
        self.time_last = 0

class LogMem:

    def __init__(self, _flock=None) -> None:
        self.data: dict = dict()
        self._plock = ProcessLock()
        self._tlock = ThreadLock()
        self._flock = _flock

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            d: dict = self.data
            for i in idx:
                assert isinstance(i, str)
                if d is None:
                    return None
                d = d.get(i, None)
            return d
        else:
            assert isinstance(idx, str)
            return self.data[idx]

    def __setitem__(self, idx, value):
        if self._flock:
            self._flock.acquire()
        with self._plock, self._tlock:
            d = self.data
            if isinstance(idx, tuple):
                for i in idx[:-1]:
                    assert isinstance(i, str)
                    if i not in d:
                        d[i] = dict()
                    if i in d and (not isinstance(d[i], dict)):
                        raise IndexError('warning: logger traj is covered')
                    d = d[i]
                assert isinstance(idx[-1], str)
                d[idx[-1]] = value
            else:
                assert isinstance(idx, str)
                d[idx] = value
        if self._flock:
            self._flock.release()

class LogJsonIdxs:

    def __init__(self, name, _flock=None, refresh=True, mode='r', backup=True) -> None:
        self.data: dict = dict()
        self._file = name if '/' in name or '\\' in name else os.path.join(log_dir(), name) + '.json'
        self._plock = ProcessLock()
        self._tlock = ThreadLock()
        self._flock = _flock
        self._refresh = refresh
        self._mode = mode
        self._backup = backup
        self._backfile = self._file + '.bk'
        check_dir(self._file)
        if 'r' in self._mode:
            self.load()

    def save(self):
        assert 'w' in self._mode
        with open(self._file, 'w') as f:
            json.dump(self.data, f)
        if self._backup:
            with open(self._backfile, 'w') as f:
                json.dump(self.data, f)

    def load(self):
        assert 'r' in self._mode
        if os.path.exists(self._file):
            try:
                with open(self._file, 'r') as f:
                    self.data = json.load(f)
            except Exception as e:
                pass

    def __getitem__(self, idx):
        x = self.__getitem(idx)
        if x is not None:
            return x
        if self._refresh:
            if self._flock:
                self._flock.acquire()
            with self._plock:
                with self._tlock:
                    self.load()
            if self._flock:
                self._flock.release()
        return self.__getitem(idx)

    def __getitem(self, idx):
        if isinstance(idx, tuple):
            d: dict = self.data
            for i in idx:
                assert isinstance(i, str)
                if d is None:
                    return None
                d = d.get(i, None)
            return d
        else:
            assert isinstance(idx, str)
            return self.data[idx]

    def __setitem__(self, idx, value):
        if self._flock:
            self._flock.acquire()
        with self._plock, self._tlock:
            if self._refresh:
                self.load()
            d = self.data
            if isinstance(idx, tuple):
                for i in idx[:-1]:
                    assert isinstance(i, str)
                    if i not in d:
                        d[i] = dict()
                    if i in d and (not isinstance(d[i], dict)):
                        raise IndexError('warning: logger traj is covered')
                    d = d[i]
                assert isinstance(idx[-1], str)
                d[idx[-1]] = value
            else:
                assert isinstance(idx, str)
                d[idx] = value
            if self._refresh:
                self.save()
        if self._flock:
            self._flock.release()

class LogToTxt:

    def __init__(self, file, span=',', new_file=True, _flock=None, csv=False, prefix=True) -> None:
        """
        :param file: "a:/1.txt" or "1"
        :param span:
        :param new_file: True: delete existing file, False: append
        :param _flock:
        :param csv: file endswith .csv, span set to ,
        :param prefix: log time at head of each line
        """
        self.__file = file if '/' in file or '\\' in file else os.path.join(log_dir(), file) + ('.csv' if csv else '.log')
        self.__span = ', ' if csv else span
        self.__prefix = prefix
        self._plock = ProcessLock()
        self._tlock = ThreadLock()
        self._flock = _flock
        if self._flock:
            self._flock.acquire()
        check_dir(self.__file)
        if new_file:
            with self._plock:
                with self._tlock:
                    if os.path.exists(self.__file):
                        os.remove(self.__file)
        if self._flock:
            self._flock.release()

    def __call__(self, *args, head=None) -> Any:
        heads = [head] if head else []
        if self.__prefix is not None:
            if self.__prefix != False:
                heads.append(now_date_time())
            if isinstance(self.__prefix, str):
                heads.append(self.__prefix)
        head = self.__span.join(heads)
        head += self.__span if len(head) > 0 else ''
        if self._flock:
            self._flock.acquire()
        with self._plock:
            with self._tlock:
                with open(self.__file, 'a', encoding='utf-8') as f:
                    f.write(head + self.__span.join([str(i) for i in args]) + '\n')
                    print(head + self.__span.join([str(i) for i in args]))
        if self._flock:
            self._flock.release()

    def print(self, *arg):
        self(arg)

    def error(self, *arg):
        self(arg, head='ERROR')

    def warning(self, *arg):
        self(arg, head='Warning')

    def todo(self, *arg):
        self(arg, head='ToDo')

class LogSimple:

    def __init__(self, file='log.log', newfile=True) -> None:
        self.__file = file
        check_dir(self.__file)
        if newfile and os.path.exists(self.__file):
            os.remove(self.__file)

    def __call__(self, *args) -> Any:
        print(', '.join([str(i) for i in args]))
        with open(self.__file, 'a', encoding='utf-8') as f:
            f.write(', '.join([str(i) for i in args]) + '\n')

    def print(self, *arg):
        self(arg)

    def log(self, *arg):
        self(arg)

class GroupLogS:

    def __init__(self, basedir='./logs'):
        self.basedir = basedir
        self.logs = [LogSimple(os.path.join(self.basedir, 'log') + '.log', newfile=False)]

    def set(self, runningname: str, newfile=True):
        self.logs.append(LogSimple(os.path.join(self.basedir, runningname) + '.log', newfile=newfile))

    def unset(self):
        if len(self.logs) > 1:
            self.logs.pop()

    def __call__(self, *args: Any):
        self.logs[-1](*args)

class LogBinIdxs(LogJsonIdxs):

    def __init__(self, name, _flock=None, refresh=True, mode='rw') -> None:
        self.data: dict = dict()
        self._file = name if '/' in name or '\\' in name else os.path.join(log_dir(), name) + '.pk'
        self._plock = ProcessLock()
        self._tlock = ThreadLock()
        self._flock = _flock
        self._refresh = refresh
        self._mode = mode
        check_dir(self._file)
        if 'r' in self._mode:
            self.load()

    def save(self):
        save(self._file, self.data)
        if self._backup:
            save(self._backfile, self.data)

    def load(self):
        if os.path.exists(self._file):
            try:
                self.data = load(self._file)
            except Exception as e:
                pass
if __name__ == '__main__':
    ld = LogBinIdxs('taabb')
    ld[1] = 2
    ld['1', (1.2, 6)] = [1, 2, 3]
    print(ld.data)

class mPrintCapturer:

    def __init__(self, out_err='out', callback=None, call_back_error=None):
        assert out_err in ['out', 'err'], 'un support type'
        self.__out_err = out_err
        self.__back = None
        self.t = ''
        self.__call = callback
        self.__call_error = call_back_error

    def __enter__(self):
        self.replace()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.restore()

    def replace(self):
        if self.__out_err == 'out':
            self.__back = sys.stdout
            sys.stdout = self
        elif self.__out_err == 'err':
            self.__back = sys.stderr
            sys.stderr = self

    def write(self, t):
        self.t += t

    def flush(self):
        t = self.t
        self.t = ''
        return t

    def restore(self):
        if self.__out_err == 'out':
            sys.stdout = self.__back
        elif self.__out_err == 'err':
            sys.stderr = self.__back

    def __del__(self):
        self.restore()

def print_table(data_lines, col_names, col_span='\t ', table_name=''):
    n_col = len(col_names)
    for d in data_lines:
        assert n_col == len(d)
    max_len = [len(str(i)) for i in col_names]
    for ci, col in enumerate(T(data_lines)):
        max_len[ci] = max(max_len[ci], max([len(str(x)) for x in col]))
    tmp = '<:' + f'>{col_span}<:'.join([str(x) for x in max_len]) + '>'
    fmt = tmp.replace('<', '{').replace('>', '}')
    print('\n' + table_name)
    print(fmt.format(*col_names))
    for row in data_lines:
        print(fmt.format(*row))

def tb_rank(data: np.ndarray, row_names, col_names, col_dim, row_dim, title: str='', round_len=None, tb_span='\t', dosum=True):
    """对row_dim做rank, 合并不remain的dim， sum rank ;
 data是值越小rank越大"""
    sp = data.shape
    assert len(sp) and col_dim != row_dim and (col_dim < len(sp)) and (row_dim < len(sp))
    assert sp[row_dim] == len(row_names) and sp[col_dim] == len(col_names)
    tb_data = np.zeros((sp[row_dim], sp[col_dim] + 1))
    if len(sp) > 2:
        dim_ranks = np.zeros_like(data)
        X = multi(sp) / sp[row_dim]
        bases = list(sp)
        del bases[row_dim]
        _idx2 = [i for i in range(sp[row_dim])]
        x = 0
        while x < X:
            idx = to_base(x, bases)
            idx1 = tuple(idx)
            idx.insert(row_dim, _idx2)
            idx = tuple(idx)
            ri = ranki_same(-data[idx])
            dim_ranks[idx] = ri
            x += 1
        X = sp[col_dim] * sp[row_dim]
        idx_all = [list(range(x)) for x in sp]
        base2 = [sp[col_dim], sp[row_dim]]
        mean_dim = [i for i in range(len(sp)) if i != col_dim and i != row_dim]
        x = 0
        while x < X:
            x1, x2 = to_base(x, base2)
            idx_all[col_dim] = x1
            idx_all[row_dim] = x2
            if len(mean_dim) > 1:
                m = np.mean(mdidx(dim_ranks, idx_all, True), axis=tuple(mean_dim))
            else:
                m = np.mean(dim_ranks[tuple(idx_all)])
            m = m.reshape(-1)
            assert len(m) == 1
            tb_data[x2, x1] = m[0]
            x += 1
    elif row_dim == 0 and col_dim == 1:
        tb_data[:, :-1] = data
    elif row_dim == 1 and col_dim == 0:
        tb_data[:, :-1] = data.T
    else:
        raise RuntimeError('Bad Dim')
    for ci in range(len(col_names)):
        tb_data[:, ci] = ranki_same(tb_data[:, ci])
    s = np.sum(tb_data, axis=1)
    tb_data[:, -1] = ranki_same(s)
    tb_head = ('', *col_names, 'Summary')
    if not dosum:
        tb_data = tb_data[:, :-1]
        tb_head = tb_head[:, :-1]
    table = []
    for ri in range(len(row_names)):
        table.append([row_names[ri], *[tb_data[ri, ci] if round_len is None else round(tb_data[ri, ci], round_len) for ci in range(len(col_names) + 1)]])
    print_table(table, tb_head, table_name=title, col_span=tb_span)

def pttb_value_rank(data: np.ndarray, row_names, col_names, col_rw: list | int, title='', round_len=None, sum_rank=True, tb_span='\t', dosum=True, dorank=True, latex12=False):
    """data:row*col col_rw[col]= -1: 值越小rank越小 1:值越小rank越大  sum_rank 是:最后一列是rank 否:sum和rank两列"""
    sp = data.shape
    assert len(sp) == 2 and sp[0] == len(row_names) and (sp[1] == len(col_names))
    if isinstance(col_rw, list):
        assert sp[1] == len(col_rw)
    else:
        col_rw = [col_rw] * sp[1]
    assert set(map(abs, col_rw)) == {1}
    tb_rank = np.zeros((sp[0], sp[1] + 1))
    for ci in range(len(col_names)):
        tb_rank[:, ci] = ranki_same(data[:, ci] * col_rw[ci])
    if dorank:
        tb_head = ['', *flatten_fast([[c, 'v&r'] for c in col_names]), 'Sum']
    else:
        tb_head = ['', *col_names, 'Sum']
    if not dosum:
        tb_head = tb_head[:-1]
    if sum_rank:
        s = np.sum(tb_rank, axis=1)
        tb_rank[:, -1] = ranki_same(s)
    else:
        if dorank:
            tb_head.append('v&r')
        s = np.sum(data, axis=1)
        tb_rank[:, -1] = ranki_same(s * col_rw[0])

    def pad0(x):
        s = str(x)
        if '.' not in s:
            return s
        return s + '0' * (round_len - len(s.split('.')[1]))
    table = []
    for ri in range(len(row_names)):
        line = [row_names[ri]]
        for ci in range(len(col_names)):
            line.append(data[ri, ci] if round_len is None else pad0(round(data[ri, ci], round_len)))
            if dorank:
                line.append(round(tb_rank[ri, ci]))
        if dosum:
            if sum_rank:
                line.append(round(tb_rank[ri, -1]))
            else:
                line.append(pad0(round(data[ri].mean(), round_len)))
                if dorank:
                    line.append(round(tb_rank[ri, -1]))
        table.append(line)

    def f1(ci, ri):
        table[ri][ci + 1] = '\\textbf{' + str(table[ri][ci + 1]) + '}'

    def f2(ci, ri):
        table[ri][ci + 1] = '\\underbar{' + str(table[ri][ci + 1]) + '}'
    if latex12:
        if dorank:
            raise RuntimeError('latex12 can not dorank')
        for ci in range(len(col_names) + (1 if dosum else 0)):
            r1 = np.argwhere(tb_rank[:, ci] == 1).reshape(-1)
            # assert len(r1) > 0
            if len(r1) <= 0:
                print(data)
                raise RuntimeError()
            if len(r1) > 1:
                for ri in r1:
                    f1(ci, ri)
                continue
            f1(ci, r1[0])
            r2 = np.argwhere(tb_rank[:, ci] == 2).reshape(-1)
            assert len(r2) > 0
            if len(r2) > 1:
                for ri in r2:
                    f2(ci, ri)
                continue
            f2(ci, r2[0])
        for ri in range(len(row_names)):
            table[ri][-1] = f'{table[ri][-1]}\\\\'
    print_table(table, tb_head, table_name=title, col_span=tb_span)