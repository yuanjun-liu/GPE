import sys, os
from traj.GPE.GPE import GPE
sys.path.extend(['./', '../', '../../'])
from _nn.nData import auto_device
import torch,math
PI = torch.pi
import torch.nn as nn
import numpy as np
from torch import Tensor

def _parse_name(name:str):
    ss=name.split('_')
    e1=float(ss[1])
    return e1

class bGPEInc(nn.Module):
    def __init__(self, name,dim=128, device=auto_device(),base_info=False) -> None:
        """dim>=4"""
        super(bGPEInc, self).__init__()
        self.dim = dim
        assert dim % 4 == 0
        self.base_i_len={1:1} 
        e=_parse_name(name)
        ' multi base '
        jz_len = [1]
        jz_base = [1]
        self.ws = [torch.ones(1).to(device) / 180 * PI]
        jz_power = np.zeros(dim)
        jz_power[0] = 1
        p = 1
        while True:
            # while jz_power[p - 1]:
            p += 1
            for j in range(int(np.log(dim) / np.log(p))):
                jz_power[p ** j - 1] = 1
            le = int(np.ceil(-np.log(e) / np.log(p)))
            le = max(le, 1)
            le = min(le, dim // 4 - sum(jz_len))
            self.ws.append(Tensor([p ** (i + 1) for i in range(le)]).to(device) / 180 * PI)
            if base_info:self.base_i_len[p]=len(self.ws[-1])
            jz_len.append(le)
            jz_base.append(p)
            if sum(jz_len) >= dim // 4:
                break

    def forward(self, T: Tensor):
        """T.shape=(n,2) or (bs,n,2)"""
        bs = 0 if len(T.shape) == 2 else len(T)
        if bs:
            T = T.reshape(-1, len(T[0][0]))
        lat, lon = (T[:, 0], T[:, 1])
        wlats, wlons = ([], [])
        for w in self.ws:
            wlats.append(lat.unsqueeze(1).expand([len(lat), len(w)]) * w)
            wlons.append(lon.unsqueeze(1).expand([len(lon), len(w)]) * w)
        wlons, wlats = (torch.concat(wlons, dim=-1), torch.concat(wlats, dim=-1))
        sc = torch.concat([torch.sin(wlons), torch.cos(wlons), torch.sin(wlats), torch.cos(wlats)], dim=-1)
        if bs:
            sc = sc.reshape(bs, -1, self.dim)
        return sc

class bGPESingle(nn.Module):
    def __init__(self, name,dim=128, device=auto_device(),base_info=False) -> None:
        """dim>=4"""
        super(bGPESingle, self).__init__()
        self.dim = dim
        assert dim % 4 == 0
        self.base_i_len={1:1} 
        ' single base '
        self.w=Tensor([2 ** i for i in range(dim//4)]).to(device) / 180 * PI
        self.num_inf=126 # torch.sum(torch.isinf(self.w))+2 -> 74 when dim=800
        self.w[self.num_inf:]=0

    def forward(self, T: Tensor):
        """T.shape=(n,2) or (bs,n,2)"""
        bs = 0 if len(T.shape) == 2 else len(T)
        if bs: T = T.reshape(-1, len(T[0][0]))
        lat, lon = (T[:, 0], T[:, 1])
        wlons = lat.unsqueeze(1).expand([len(lat), len(self.w)]) * self.w
        wlats = lon.unsqueeze(1).expand([len(lon), len(self.w)]) * self.w
        slon,clon,sla,cla=torch.sin(wlons), torch.cos(wlons), torch.sin(wlats), torch.cos(wlats)
        slon[:,self.num_inf:]=0;clon[:,self.num_inf:]=0;sla[:,self.num_inf:]=0;cla[:,self.num_inf:]=0
        sc = torch.concat([slon,clon,sla,cla], dim=-1)
        if bs: sc = sc.reshape(bs, -1, self.dim)
        return sc


class bGPEPrime(nn.Module):
    def __init__(self,name, dim=128, device=auto_device(),base_info=False) -> None:
        """dim>=4"""
        super(bGPEPrime, self).__init__()
        self.dim = dim
        assert dim % 4 == 0
        self.base_i_len={1:1} 
        e=_parse_name(name)
        primes=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1123, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571]
        jz_len = [1]
        jz_base = [1]
        self.ws = [torch.ones(1).to(device) / 180 * PI]
        jz_power = np.zeros(dim)
        jz_power[0] = 1
        p = 1
        while True:
            p =primes[0] ; del primes[0]
            for j in range(int(np.log(dim) / np.log(p))):
                jz_power[p ** j - 1] = 1
            le = int(np.ceil(-np.log(e) / np.log(p)))
            le = max(le, 1)
            le = min(le, dim // 4 - sum(jz_len))
            self.ws.append(Tensor([p ** (i + 1) for i in range(le)]).to(device) / 180 * PI)
            if base_info:self.base_i_len[p]=len(self.ws[-1])
            jz_len.append(le)
            jz_base.append(p)
            if sum(jz_len) >= dim // 4:
                break

    def forward(self, T: Tensor):
        """T.shape=(n,2) or (bs,n,2)"""
        bs = 0 if len(T.shape) == 2 else len(T)
        if bs:
            T = T.reshape(-1, len(T[0][0]))
        lat, lon = (T[:, 0], T[:, 1])
        wlats, wlons = ([], [])
        for w in self.ws:
            wlats.append(lat.unsqueeze(1).expand([len(lat), len(w)]) * w)
            wlons.append(lon.unsqueeze(1).expand([len(lon), len(w)]) * w)
        wlons, wlats = (torch.concat(wlons, dim=-1), torch.concat(wlats, dim=-1))
        sc = torch.concat([torch.sin(wlons), torch.cos(wlons), torch.sin(wlats), torch.cos(wlats)], dim=-1)
        if bs:
            sc = sc.reshape(bs, -1, self.dim)
        return sc


class bGPEAtt(nn.Module): 
    def __init__(self, name,dim=128, device=auto_device(),base_info=False) -> None:
        """dim>=4"""
        super(bGPEAtt, self).__init__()
        self.dim = dim
        assert dim % 4 == 0
        self.w=torch.exp(torch.arange(0, dim//2, 2) * (-math.log(10000)) / dim//2).to(device)/180*PI 

    def forward(self, T: Tensor):
        """T.shape=(n,2) or (bs,n,2)"""
        bs = 0 if len(T.shape) == 2 else len(T)
        if bs:
            T = T.reshape(-1, len(T[0][0]))
        lat, lon = (T[:, 0], T[:, 1])
        wlats, wlons = ([], [])
        for w in [self.w]:
            wlats.append(lat.unsqueeze(1).expand([len(lat), len(w)]) * w)
            wlons.append(lon.unsqueeze(1).expand([len(lon), len(w)]) * w)
        wlons, wlats = (torch.concat(wlons, dim=-1), torch.concat(wlats, dim=-1))
        sc = torch.concat([torch.sin(wlons), torch.cos(wlons), torch.sin(wlats), torch.cos(wlats)], dim=-1)
        if bs:
            sc = sc.reshape(bs, -1, self.dim)
        return sc


def _base_raw_gpe(name,dim):
    base_i_len={1:1} 
    e=_parse_name(name)
    assert 0<e <= 1
    ' multi base '
    jz_len = [1]
    jz_base = [1]
    ws = [torch.ones(1) / 180 * PI]
    jz_power = np.zeros(dim)
    jz_power[0] = 1
    p = 1
    while True:
        while jz_power[p - 1]:
            p += 1
        # for j in range(int(np.log(dim) / np.log(p))):
        #     jz_power[p ** j - 1] = 1
        for j in range(dim):
            jzp=p**j-1
            if jzp>=dim:break
            jz_power[jzp]=1
        le = int(np.ceil(-np.log(e) / np.log(p)))
        le = max(le, 1)
        le = min(le, dim // 4 - sum(jz_len))
        ws.append(Tensor([p ** (i + 1) for i in range(le)]) / 180 * PI)
        base_i_len[p]=len(ws[-1])
        jz_len.append(le)
        jz_base.append(p)
        if sum(jz_len) >= dim // 4:
            break
    return base_i_len


def gpe_bases_info(name,dim):
    if 'bGPEInc' in name: 
        gpe=bGPEInc(name,dim,base_info=True)
        return gpe.base_i_len
    elif 'bGPEPrime' in name:
        gpe=bGPEPrime(name,dim,base_info=True)
        return gpe.base_i_len
    elif 'bGPESingle' in name:
        gpe=bGPESingle(name,dim,base_info=True)
        return gpe.base_i_len
    elif 'bGPEAtt' in name:return {}
    elif 'bGPE_' in name:
        return _base_raw_gpe(name,dim)
    else: raise RuntimeError('bad gpe name')



def load_gpe_others(name,device=auto_device(), dim=128, **kw):
    if 'bGPEInc' in name: cls=bGPEInc
    elif 'bGPEPrime' in name:cls=bGPEPrime
    elif 'bGPEAtt' in name:cls=bGPEAtt
    elif 'bGPESingle' in name:cls=bGPESingle
    elif 'bGPE_' in name:cls=GPE
    else: raise RuntimeError('bad gpe name')
    gpe = cls(name=name, dim=dim, device=device)
    gpe.requires_grad_(False)
    return gpe
