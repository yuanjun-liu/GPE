import functools
import time
from functools import wraps
from _tool.mData import isviadecorator
from datetime import datetime
fmt_day = '%Y-%m-%d'
fmt_hms = '%H:%M:%S'

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


def now_date_time():
    return time.strftime(fmt_day + ' ' + fmt_hms, time.localtime())
