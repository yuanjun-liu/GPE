import functools
import time
from functools import wraps
from _tool.mData import isviadecorator
from datetime import datetime
fmt_day = '%Y-%m-%d'
fmt_hms = '%H:%M:%S'

def tim2stamp(t: str):
    """t:2011-01-01 13:12:49 like"""
    return time.mktime(time.strptime(t, fmt_day + ' %X'))

def tim2sec(tim):
    import re
    if re.match('[0-9]+[dshm]', tim):
        t, f = (float(tim[:-1]), tim[-1])
        if f == 'd':
            f = 'h'
            t *= 24
        if f == 'h':
            f = 'm'
            t *= 60
        if f == 'm':
            f = 's'
            t *= 60
        if f == 's':
            return int(t)
    if tim.endswith('am') or tim.endswith('pm'):
        if tim.endswith('pm'):
            tim = tim.replace('pm', '')
            add = 12
        else:
            add = 0
            tim = tim.replace('am', '')
        tt = tim.split(' ')
        t2t = tt[1].split(':')
        if int(t2t[0]) == 12 and add == 12:
            add = 0
        if len(t2t) > 3:
            t2t = t2t[:3]
        if '.' in t2t[2]:
            t2t[2] = t2t[2][:t2t[2].index('.')]
        tim = tt[0] + ' ' + str(int(t2t[0]) + add) + ':' + ':'.join(t2t[1:])
    if re.match('\\d{4}-\\d+-\\d+ \\d+:\\d+:\\d+.*', tim):
        t = time.strptime(tim, '%Y-%m-%d %X')
        return int(time.mktime(t))
    if re.match('\\d{4}/\\d+/\\d+ \\d+:\\d+:\\d+.*', tim):
        t = time.strptime(tim, '%Y/%m/%d %X')
        return int(time.mktime(t))
    assert False, 'convert to time false'

def mon2sint(mon: str):
    d = {'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 'may': '05', 'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12'}
    return d[mon[:3].lower()]

def watch_time(func=None, *arg, **kw):
    """return val, tim"""

    def decorator(func):

        @wraps(func)
        def wrapped_function(*args, **kwargs):
            t1 = time.time()
            res = func(*args, **kwargs)
            t2 = time.time()
            return (res, t2 - t1)
        return wrapped_function
    if isviadecorator():
        return decorator
    else:
        t1 = time.time()
        res = func(*arg, **kw)
        return (res, time.time() - t1)

class WatchTime:

    def __init__(self, name='', fmt='{:.3f}(s)', fun=None) -> None:
        self.t1 = 0
        self.name = name
        self.fmt = fmt
        self.fun = fun

    def __enter__(self):
        self.t1 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t2 = time.time()
        t = t2 - self.t1
        if self.fun:
            t = self.fun(t)
        print(f'{self.name}用时：{self.fmt}'.format(t))

class PrintTime:

    def __init__(self, name='', fmt='{:.3f}(s)') -> None:
        self.t1 = 0
        self.name = name
        self.fmt = fmt

    def __enter__(self):
        self.t1 = time.process_time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        t2 = time.process_time()
        t = t2 - self.t1
        print(f'{self.name}用时：{self.fmt}'.format(t))

def run_mutil_times(times, func, *arg, **kw):
    res = []
    for i in range(times):
        res.append(func(*arg, **kw))
    return res

def now_date():
    return time.strftime(fmt_day, time.localtime())

def now_time():
    return time.strftime(fmt_hms, time.localtime())

def now_date_time():
    return time.strftime(fmt_day + ' ' + fmt_hms, time.localtime())

def now_timestamp():
    return int(time.time())

def timestamp2date(x):
    return datetime.fromtimestamp(x).strftime(fmt_day + ' ' + fmt_hms)

def stamp2ymwds(t: int):
    """year,month,day_week,day_month,day_year,sec_day"""
    d: datetime = datetime.fromtimestamp(t)
    sec_day = d.hour * 60 * 60 + d.minute * 60 + d.second
    dt = d.timetuple()
    return [dt.tm_year, dt.tm_mon, dt.tm_wday, dt.tm_mday, dt.tm_yday, sec_day]
if __name__ == '__main__':
    pass