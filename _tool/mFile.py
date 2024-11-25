import sys, os
is_ipynb = 'ipykernel' in sys.modules
is_win = sys.platform == 'win32' or sys.platform == 'cygwin'
is_linux = sys.platform == 'linux'
is_mac = sys.platform == 'darwin'
path_base = 'a:\\KyData' if is_win else '../KyData' if is_linux else '/Volumes/SD/kyData'
out_base = 'a:\\KyOut' if is_win else './output'
path_traj = os.path.join(path_base, 'TrajData')
path_img = os.path.join(path_base, 'img')
path_map = os.path.join(path_base, 'map')
path_high_dim = os.path.join(path_base, 'HighDim')
path_nlp = os.path.join(path_base, 'NLP')

def relative_path(path=sys.argv[0]):
    if is_ipynb:
        from IPython import get_ipython
        ip = get_ipython()
        if '__vsc_ipynb_file__' in ip.user_ns:
            path = ip.user_ns['__vsc_ipynb_file__']
    return path.replace(os.getcwd(), '.').replace('\\', '/')

def packet_names():
    return '.'.join(relative_path().split('/')[:-1]).replace('..', '')

def check_dir(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def check_file(file):
    return os.path.exists(file)
check_dir(out_base)

def out_dir(name=''):
    dir = os.path.join(out_base, packet_names())
    if name != '':
        dir = os.path.join(dir, name + os.path.sep)
    check_dir(dir)
    return dir

def res_dir():
    return out_dir('res')

def log_dir():
    return out_dir('log')

def ckpt_dir():
    return out_dir('ckpt')

def cache_dir():
    return out_dir('cache')

def list_dir(path, deep=False):
    if not deep:
        for root, dirs, files in os.walk(path):
            return [dirs, files, [os.path.join(root, d) for d in dirs], [os.path.join(root, f) for f in files]]
    else:
        dir_name, file_name, dir_path, file_path = ([], [], [], [])
        for root, dirs, files in os.walk(path):
            dir_name.extend(dirs)
            file_name.extend(files)
            dir_path.extend([os.path.join(root, d) for d in dirs])
            file_path.extend([os.path.join(root, f) for f in files])
        return (dir_name, file_name, dir_path, file_path)

def parse_line(line: str, split, *type):
    ss = line.split(split)
    assert len(ss) >= len(type)
    for i in range(len(ss) - len(type)):
        type = list(type)
        type.append(type[-1])
    res = [None] * len(ss)
    for i in range(len(ss)):
        res[i] = type[i](ss[i])
    return res

def file_size(file):
    return os.stat(file).st_size

def file_time_create(file):
    return os.stat(file).st_ctime

def file_time_modify(file):
    return os.stat(file).st_mtime

def file_name(file, ex=False):
    assert os.path.exists(file)
    assert isinstance(file, str)
    if os.path.isdir(file):
        return os.path.dirname(file)
    name = file.replace('\\', '/').split('/')[-1]
    if ex:
        return name
    else:
        return '.'.join(name.split('.')[:-1])

def read_lines(f):
    with open(f, 'r', encoding='utf-8') as f:
        return f.readlines()

def copy(res, dst):
    import shutil
    assert os.path.exists(res)
    if os.path.isdir(res):
        if not os.path.exists(dst):
            os.makedirs(dst)
    elif not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))
    if os.path.isdir(res):
        for root, dirs, files in os.walk(res):
            for file in files:
                r = os.path.join(root, file)
                d = os.path.join(dst, '.' + r[len(res):])
                copy(r, d)
    else:
        shutil.copy(res, dst)

def read_lines_iter(f, skip_lines=0, encoding='utf-8', block='500m'):
    with open(f, encoding=encoding) as f:
        if block is None:
            for i in range(skip_lines):
                f.readline()
            while True:
                line = f.readline()
                if not line:
                    break
                yield line
            return
        if block == 'all':
            return f.readlines()[skip_lines:]
        if block.endswith('g'):
            block = f'{float(block[:-1]) * 1024}m'
        if block.endswith('m'):
            size = int(float(block[:-1]) * 1024 * 1024)
            buf = ''
            while True:
                ctx = f.read(size)
                if not ctx:
                    if buf:
                        return None if skip_lines > 0 else buf
                    else:
                        return
                ctx = buf + ctx
                lines = ctx.split('\n')
                if len(lines) > 0:
                    buf, lines = (lines[-1], lines[:-1])
                    if skip_lines > 0:
                        sk = min(skip_lines, len(lines))
                        skip_lines -= sk
                        lines = lines[sk:]
                    for line in lines:
                        yield (line + '\n')
        raise 'bad block'

def read_xline(file, x, skip_lines=0):
    with open(file, 'r', encoding='utf-8') as f:
        for i in range(skip_lines):
            f.readline()
        for i in range(x):
            print(f.readline())