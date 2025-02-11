import sys, os
sys.path.extend(['.', '..', '...'])
import numpy as np
from _tool.mFile import cache_dir
import torch
from torch import Tensor
from gensim.models import Word2Vec
from _nn.nData import reset_param_uniform, auto_device
from _nn.nFile import load_weight, save_weight
from _data.traj.load_trajs import traj_bbox
from _nn.nLoss import ContrastiveLossInfoNCE

class GridSpace:
    def __init__(self, bbox, nxy=None, dxy=None):
        [[xmin, xmax], [ymin, ymax]] = bbox
        self.bbox = [xmin, xmax, ymin, ymax]
        self._dmax = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2) ** 0.5
        self.w, self.h = (xmax - xmin, ymax - ymin)
        self.area = (xmax - xmin) * (ymax - ymin)
        assert nxy is not None or dxy is not None
        if nxy is not None:
            self.nx, self.ny = nxy
            self.dx, self.dy = ((xmax - xmin) / self.nx, (ymax - ymin) / self.ny)
        else:
            self.dx, self.dy = dxy
            self.nx, self.ny = (int(np.ceil((xmax - xmin) / self.dx)), int(np.ceil((ymax - ymin) / self.dy)))
        self.num_grids = int(self.nx * self.ny)

    def p2g(self, x, y):
        g = torch.Tensor((x - self.bbox[0]) / self.dx).int() + torch.Tensor(self.nx * ((y - self.bbox[2]) / self.dy)).int()
        g[(g > self.num_grids) + (g < 0)] = self.num_grids
        return g

    def g2p(self, gi):
        x, y = (gi % self.nx, gi // self.nx)
        return (self.bbox[0] + x * self.dx + self.dx / 2, self.bbox[2] + y * self.dy + self.dy / 2)

    def ts2gs(self, ts):
        return [list(self.p2g(t[:, 0], t[:, 1])) for t in ts]

    def wv_predefined_walk(self):
        """for gensim.word2vec, | - / \\"""
        res = []
        lines = [[self.nx * y + x for x in range(self.nx)] for y in range(self.ny)]
        res.extend(lines)
        res.extend([l[::-1] for l in lines])
        lines = [[self.nx * y + x for y in range(self.ny)] for x in range(self.nx)]
        res.extend(lines)
        res.extend([l[::-1] for l in lines])
        lines = [[self.nx * y + (x - y) for y in range(min(self.ny, x + 1)) if x - y < self.nx] for x in range(self.nx + self.ny + 1)]
        res.extend(lines)
        res.extend([l[::-1] for l in lines])
        lines = [[self.nx * y + (x + y) for y in range(min(self.ny, self.nx - x)) if 0 <= x + y < self.nx] for x in range(-self.ny, self.nx)] if self.nx >= self.ny else [[self.nx * (y + x) + x for x in range(self.nx) if 0 <= x + y < self.ny] for y in range(-self.nx, self.ny)]
        res.extend(lines)
        res.extend([l[::-1] for l in lines])
        return [l for l in res if len(l) > 1]

    def gs_plt(self, gss):
        import matplotlib.pyplot as plt
        if '__len__' not in gss[0].__dir__():
            gss = [gss]
        for gs in gss:
            XY = list(zip(*[self.g2p(g) for g in gs]))
            plt.plot(XY[0], XY[1], 'r.-')
        plt.show()

class GridEbd(torch.nn.Module):

    def __init__(self, dim, city, nxy=None, dxy=None, city_suffix='', **kw):
        super(GridEbd, self).__init__()
        self._path = os.path.join(cache_dir(), f'GridE_{city}{city_suffix}_{dim}_{(f'{nxy[0]}_{nxy[1]}' if nxy is not None else f'{dxy[0]}_{dxy[1]}')}')
        self.dim = dim
        self._space: GridSpace = GridSpace(traj_bbox[city], nxy, dxy)
        self.ebd = torch.nn.Embedding(self._space.num_grids + 1, dim, padding_idx=self._space.num_grids)
        self._trained = os.path.exists(self._path + '.th')
        if self._trained:
            self.__load()  
            self.requires_grad_(False)
        else: reset_param_uniform(self.ebd.weight, dim)

    def __call__(self, T, is_grid=False):
        if is_grid:
            return T
        if len(T.shape) == 2:
            return self.ebd(self._space.p2g(T[:, 0], T[:, 1]))
        return self.ebd(self._space.p2g(T[:, :, 0], T[:, :, 1]))

    def __save(self):
        save_weight(self.ebd, self._path + '.th')

    def __load(self):
        load_weight(self.ebd, self._path + '.th')

    def word2vec(self, ts=None):
        if self._trained: return
        tsg = self._space.wv_predefined_walk() if ts is None else self._space.ts2gs(ts)
        sents = [('g' + ' g'.join(map(str, t))).split() for t in tsg]
        model = Word2Vec(sents, vector_size=self.dim, workers=4,sg=1,epochs=20)
        w2 = torch.zeros((self._space.num_grids + 1, self.dim))
        reset_param_uniform(w2, self.dim)
        w = torch.Tensor(np.array([ model.wv[f'g{i}'] if f'g{i}' in model.wv else w2[i] for i in range(self._space.num_grids + 1)]))
        self.ebd = torch.nn.Embedding.from_pretrained(w)
        self.__save()


class GxyEbd(torch.nn.Module):

    def __init__(self, dim, city, nxy=None, dxy=None, city_suffix='', **kw):
        super(GxyEbd, self).__init__()
        self._path = os.path.join(cache_dir(), f'GxyE_{city}{city_suffix}_{dim}_{(f'{nxy[0]}_{nxy[1]}' if nxy is not None else f'{dxy[0]}_{dxy[1]}')}')
        self.dim = dim
        _space: GridSpace = GridSpace(traj_bbox[city], nxy, dxy)
        self.bbox = _space.bbox
        self.nx = _space.nx
        self.ny = _space.ny
        self.dx = _space.dx
        self.dy = _space.dy
        self.ebdx = torch.nn.Embedding(self.nx + 1, dim, padding_idx=self.nx)
        self.ebdy = torch.nn.Embedding(self.ny + 1, dim, padding_idx=self.ny)
        self._trained = os.path.exists(self._path + '.th')
        if self._trained:
            self.__load()
            self.requires_grad_(False)
        else:
            reset_param_uniform(self.ebdx.weight, dim)
            reset_param_uniform(self.ebdy.weight, dim)

    def __call__(self, T, is_grid=False):
        if is_grid:
            return T
        if len(T.shape) == 2:
            x, y = (T[:, 0], T[:, 1])
        else:
            x, y = (T[:, :, 0], T[:, :, 1])
        x = torch.Tensor((x - self.bbox[0]) / self.dx).int()
        y = torch.Tensor((y - self.bbox[2]) / self.dy).int()
        x[(x > self.nx) + (x < 0)] = self.nx
        y[(y > self.ny) + (y < 0)] = self.ny
        return self.ebdx(x) + self.ebdy(y)

    def __save(self):
        save_weight(self, self._path + '.th')

    def __load(self):
        load_weight(self, self._path + '.th')

    def pretrain(self):
        if self._trained:
            return
        e_stop = 100
        bs = 1024
        device = auto_device()
        posss = {(i, j): ([(i - 1, j - 1)] if i > 0 and j > 0 else []) + ([(i - 1, j)] if i > 0 else []) + ([(i, j - 1)] if j > 0 else []) + ([(i + 1, j + 1)] if i < self.nx - 1 and j < self.ny - 1 else []) + ([(i + 1, j)] if i < self.nx - 1 else []) + ([(i, j + 1)] if j < self.ny - 1 else []) for i in range(self.nx) for j in range(self.ny)}

        def dataset():
            while True:
                ps = [[np.random.randint(self.nx), np.random.randint(self.ny)] for _ in range(bs)]
                ps2 = [posss[x, y][np.random.randint(len(posss[x, y]))] for x, y in ps]
                yield (torch.LongTensor(ps).to(device), torch.LongTensor(ps2).to(device))
        model = self.to(device)
        model.train()
        lossf = ContrastiveLossInfoNCE(bs)
        opt = torch.optim.AdamW(model.parameters())
        best_i, best_loss = (0, 999999)
        for bi, (ps, ps2) in enumerate(dataset()):
            v1 = self.ebdx(ps[:, 0]) + self.ebdy(ps[:, 1])
            v2 = self.ebdx(ps2[:, 0]) + self.ebdy(ps2[:, 1])
            loss: Tensor = lossf(v1, v2)
            lossv = loss.item()
            if bi - best_i > e_stop:
                break
            if lossv < best_loss:
                best_i, best_loss = (bi, lossv)
                self.__save()
                print(f'batch:{bi}, loss:{lossv}')
            opt.zero_grad()
            loss.backward()
            opt.step()
        model = model.cpu()
        self.__save()

