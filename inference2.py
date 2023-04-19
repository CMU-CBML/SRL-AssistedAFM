import numpy as np
from numpy import cos, sin, array, argmin, argmin, zeros, full, delete, concatenate, save, equal, where, power, mean, any
from numpy import arccos as acos
from numpy import max as nmax
from numpy import min as nmin
from numpy import maximum as nmax2
from numpy import minimum as nmin2
import torch
import torch.nn as nn
import torch.utils.data as data
import time
from torch.nn import functional as F
from numba import jit
import matplotlib.pyplot as plt

iga = 0
max_episode = 100000
max_point = 100000
hl = 1000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class resnet0(nn.Module):
    def __init__(self):
        super(resnet0, self).__init__()
        self.linear1 = nn.Linear(23, hl)
        self.linear2 = nn.Linear(hl, hl)
        self.linear3 = nn.Linear(hl, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        for i in range(50):
            residual = out
            out = self.linear2(out)
            out = self.relu(out)
            out = self.linear2(out)
            out += residual
            out = self.relu(out)
        out = self.linear3(out)
        return out

class resnet1(nn.Module):
    def __init__(self):
        super(resnet1, self).__init__()
        self.linear1 = nn.Linear(23, hl)
        self.linear2 = nn.Linear(hl, hl)
        self.linear3 = nn.Linear(hl, 4)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        for i in range(50):
            residual = out
            out = self.linear2(out)
            out = self.relu(out)
            out = self.linear2(out)
            out += residual
            out = self.relu(out)
        out = self.linear3(out)
        return out

class resnet2(nn.Module):
    def __init__(self):
        super(resnet2, self).__init__()
        self.linear1 = nn.Linear(23, hl)
        self.linear2 = nn.Linear(hl, hl)
        self.linear3 = nn.Linear(hl, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        for i in range(50):
            residual = out
            out = self.linear2(out)
            out = self.relu(out)
            out = self.linear2(out)
            out += residual
            out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        return out

class resnet3(nn.Module):
    def __init__(self):
        super(resnet3, self).__init__()
        self.linear1 = nn.Linear(23, hl)
        self.linear2 = nn.Linear(hl, hl)
        self.linear3 = nn.Linear(hl, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        for i in range(50):
            residual = out
            out = self.linear2(out)
            out = self.relu(out)
            out = self.linear2(out)
            out += residual
            out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        return out

classification = resnet0().to(device)
classification.load_state_dict(torch.load('classification.pth'))
fourclassification = resnet1().to(device)
fourclassification.load_state_dict(torch.load('fourclassification2.pth'))
regression = resnet2().to(device)
regression.load_state_dict(torch.load('regression.pth'))
regression2 = resnet3().to(device)
regression2.load_state_dict(torch.load('regression2.pth'))

@jit(nopython = True)
def build_input_tensor(dataset1, coordinate, polygon, info, lp, connect_count, initnum, total_length):
    _360deg = 2 * np.pi
    _90deg = 0.5 * np.pi
    j = argmin(info[:lp, 1] + info[:lp, 3] * lp / 12 / total_length)
    jm1 = (j - 1) % lp
    jm2 = (j - 2) % lp
    jm3 = (j - 3) % lp
    jp1 = (j + 1) % lp
    jp2 = (j + 2) % lp
    x, y = coordinate[polygon[j], :]
    xr1, yr1 = coordinate[polygon[jm1], :]
    xr2, yr2 = coordinate[polygon[jm2], :]
    xr3, yr3 = coordinate[polygon[jm3], :]
    xl1, yl1 = coordinate[polygon[jp1], :]
    xl2, yl2 = coordinate[polygon[jp2], :]
    xl3, yl3 = coordinate[polygon[(j + 3) % lp], :]
    xr4, yr4 = coordinate[polygon[(j - 4) % lp], :]
    xl4, yl4 = coordinate[polygon[(j + 4) % lp], :]
    ba = info[j, 0]
    c, s = xr1 - x, yr1 - y
    k = (j + 5) % lp
    distn1 = (x - coordinate[polygon[k], 0]) ** 2 + (y - coordinate[polygon[k], 1]) ** 2
    num1 = k
    if lp > 10:
        infotmp = info[jm1, 2]
        for kcount in range(lp - 9):
            xout, yout = coordinate[polygon[k], 0] - x, coordinate[polygon[k], 1] - y
            dist = xout ** 2 + yout ** 2
            if dist < distn1 and dist > 0:
                angle = acos(nmax2(-1, nmin2(1, (xout * c + yout * s) / infotmp * dist ** -0.5)))
                if c * yout < xout * s:
                    angle = _360deg - angle
                if angle < ba:
                    distn1 = dist
                    num1 = k
            k = (k + 1) % lp
    xn1, yn1, d = coordinate[polygon[num1], 0] - x, coordinate[polygon[num1], 1] - y, c ** 2 + s ** 2
    c /= d
    s /= d
    reference_length = info[j, 3]
    dataset1[0, 0: 17] = c * (xr2 - x) + s * (yr2 - y), s * (x - xr2) + c * (yr2 - y), c * (xr3 - x) + s * (yr3 - y), s * (x - xr3) + c * (yr3 - y), c * (xl1 - x) + s * (yl1 - y), s * (x - xl1) + c * (yl1 - y), c * (xl2 - x) + s * (yl2 - y), s * (x - xl2) + c * (yl2 - y), c * (xl3 - x) + s * (yl3 - y), s * (x - xl3) + c * (yl3 - y), c * (xr4 - x) + s * (yr4 - y), s * (x - xr4) + c * (yr4 - y), c * (xl4 - x) + s * (yl4 - y), s * (x - xl4) + c * (yl4 - y), c * xn1 + s * yn1, -s * xn1 + c * yn1, reference_length * lp / 12 / total_length
    dataset1[0, 14: 16] *= nmin2(1, reference_length / (xn1 ** 2 + yn1 ** 2) ** 0.5)
    if polygon[jm2] < initnum:
        dataset1[0, 17] = connect_count[polygon[jm2]] - 2
    else:
        dataset1[0, 17] = connect_count[polygon[jm2]] - 3
    if polygon[jm1] < initnum:
        dataset1[0, 18] = connect_count[polygon[jm1]] - 2
    else:
        dataset1[0, 18] = connect_count[polygon[jm1]] - 3
    if polygon[j] < initnum:
        dataset1[0, 19] = connect_count[polygon[j]] - 2
    else:
        dataset1[0, 19] = connect_count[polygon[j]] - 3
    if polygon[jp1] < initnum:
        dataset1[0, 20] = connect_count[polygon[jp1]] - 2
    else:
        dataset1[0, 20] = connect_count[polygon[jp1]] - 3
    if polygon[jp2] < initnum:
        dataset1[0, 21] = connect_count[polygon[jp2]] - 2
    else:
        dataset1[0, 21] = connect_count[polygon[jp2]] - 3
    return dataset1, c, s, x, y, xr1, yr1, xr2, yr2, xr3, yr3, xl1, yl1, xl2, yl2, xl3, yl3, xl4, yl4, xr4, yr4, j, reference_length
#@jit
def implementation(initnum, coordinate, polygon, info, max_episode, max_point, total_length, iga):
    _360deg = 2 * np.pi
    _90deg = 0.5 * np.pi
    _45deg = 0.25 * np.pi
    _30deg = np.pi / 6
    quad = zeros((max_episode, 4), dtype = int)
    connect_count = full(max_point, 2, dtype = int)
    dataset1 = full((1, 23), iga, dtype = float)
    connect = zeros((max_point, max_point), dtype = bool)
    connect[0, initnum - 1], connect[0, 1], connect[initnum - 1, initnum - 2], connect[initnum - 1, 0] = True, True, True, True
    for i in range(1, initnum - 1):
        connect[i, i - 1], connect[i, i + 1] = True, True
    connect_len = initnum - 1
    lp = initnum
    with torch.no_grad():
        for i in range(max_episode):
            for j in range(lp):
                dataset1, c, s, ox, oy, rx, ry, r2x, r2y, r3x, r3y, lx, ly, l2x, l2y, l3x, l3y, l4x, l4y, r4x, r4y, k, reference_length = build_input_tensor(dataset1, coordinate, polygon, info, lp, connect_count, initnum, total_length)
                if torch.argmax(F.softmax(classification(torch.FloatTensor(dataset1).to(device)), dim = -1), dim = -1) == 0:
                    break
                info[k, 1] = 114514
            update_type = torch.argmax(F.softmax(fourclassification(torch.FloatTensor(dataset1).to(device)), dim = -1), dim = -1)
            j = k
            jm6, jm5, jm4, jm3, jm2, jm1, jp1, jp2, jp3, jp4, jp5 = (j - 6) % lp, (j - 5) % lp, (j - 4) % lp, (j - 3) % lp, (j - 2) % lp, (j - 1) % lp, (j + 1) % lp, (j + 2) % lp, (j + 3) % lp, (j + 4) % lp, (j + 5) % lp
            l, l2, l3, l4, l5, l6, l7, o, r, r2, r3, r4, r5, r6, r7 = polygon[jp1], polygon[jp2], polygon[jp3], polygon[jp4], polygon[jp5], polygon[(j + 6) % lp], polygon[(j + 7) % lp], polygon[j], polygon[jm1], polygon[jm2], polygon[jm3], polygon[jm4], polygon[jm5], polygon[jm6], polygon[(j - 7) % lp]
            l5x, l5y = coordinate[l5, :]
            l6x, l6y = coordinate[l6, :]
            l7x, l7y = coordinate[l7, :]
            r5x, r5y = coordinate[r5, :]
            r6x, r6y = coordinate[r6, :]
            r7x, r7y = coordinate[r7, :]
#             if i % 1000 == 0:
#                 plt.figure(figsize = (30, 30))
#                 plt.axis("equal")
#                 plt.xticks([])
#                 plt.yticks([])
#                 plt.axis("off")
#                 for ii in range(i + 1):
#                     iii = quad[ii]
#                     for jj in range(4):
#                         plt.plot([coordinate[iii[jj], 0], coordinate[iii[(jj + 1) % 4], 0]], [coordinate[iii[jj], 1], coordinate[iii[(jj + 1) % 4], 1]], color = '#5F203D', linewidth = 1)
#                 plt.savefig(str(i) + '.png', dpi = 1000)
            if update_type == 0:
                raw_coordinate = regression(torch.FloatTensor(dataset1).to(device)).cpu().detach().numpy()
                if abs(info[j, 0] - _90deg) < 1e-6:
                    radius, angle = 2 ** 0.5, _45deg
                else:
                    radius, angle = raw_coordinate[0, 1] * reference_length / info[jm1, 2] / 1.2, raw_coordinate[0, 0] * info[j, 0]
                nxr, nyr = radius * cos(angle), radius * sin(angle)
                connect_count[r] += 1
                connect_count[l] += 1
                connect_len += 1
                connect_count[connect_len], connect[r, connect_len], connect[l, connect_len], connect[connect_len, r], connect[connect_len, l], polygon[j] = 2, True, True, True, True, connect_len
                sc2 = s ** 2 + c ** 2
                nx, ny = ox + (c * nxr - s * nyr) / sc2, oy + (s * nxr + c * nyr) / sc2
                
                

                if ry == ly:
                    ymid = ny
                    xmid = 0.5 * (lx + rx)
                else:
                    ymid = ((lx * ny - rx * ny + nx * ry - nx * ly) * (2 * lx - 2 * rx) - (ly ** 2 - ry ** 2 + lx ** 2 - rx ** 2) * (ry - ly)) / 2 / ((ry - ly) ** 2 + (rx - lx) ** 2)
                    xmid = ((lx * ny - rx * ny + nx * ry - nx * ly) + ymid * (rx - lx)) / (ry - ly)
                dd1 = info[jp1, 2]
                dd2 = info[jm2, 2]
                dd3 = ((nx - lx) ** 2 + (ny - ly) ** 2) ** 0.5
                dd4 = ((nx - rx) ** 2 + (ny - ry) ** 2) ** 0.5
                dd5 = ((xmid - lx) ** 2 + (ymid - ly) ** 2) ** 0.5
                dd6 = ((xmid - rx) ** 2 + (ymid - ry) ** 2) ** 0.5
                if min(dd1 / dd5, dd5 / dd1, dd5 / dd6, dd6 / dd5, dd6 / dd2, dd2 / dd6) > min(dd1 / dd3, dd3 / dd1, dd3 / dd4, dd4 / dd3, dd4 / dd2, dd2 / dd4):
                    nx = xmid
                    ny = ymid
                coordinate[connect_len, :] = nx, ny

                
                
                
                new_length1, new_length2 = ((nx - lx) ** 2 + (ny - ly) ** 2) ** 0.5, ((nx - rx) ** 2 + (ny - ry) ** 2) ** 0.5
                total_length -= info[jm1, 2] + info[j, 2] - new_length1 - new_length2
                info[jm1, 2], info[j, 2] = new_length2, new_length1
                x31, x21, y31, y21 = nx - rx, r2x - rx, ny - ry, r2y - ry
                info[jm1, 0] = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    info[jm1, 0] = _360deg - info[jm1, 0]
                x31, x21, y31, y21 = lx - nx, rx - nx, ly - ny, ry - ny
                info[j, 0] = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    info[j, 0] = _360deg - info[j, 0]
                x31, x21, y31, y21 = l2x - lx, nx - lx, l2y - ly, ny - ly
                info[jp1, 0] = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    info[jp1, 0] = _360deg - info[jp1, 0]
                x31, x21, y31, y21 = rx - r3x, r5x - r3x, ry - r3y, r5y - r3y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = nx - r3x, r6x - r3x, ny - r3y, r6y - r3y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm3, 1] = info[jm3, 0] + a1 + a2
                info[jm3, 3] = info[jm6, 2] + info[jm5, 2] + info[jm4, 2] + info[jm3, 2] + info[jm2, 2] + new_length2
                x31, x21, y31, y21 = nx - r2x, r4x - r2x, ny - r2y, r4y - r2y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = lx - r2x, r5x - r2x, ly - r2y, r5y - r2y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm2, 1] = info[jm2, 0] + a1 + a2
                info[jm2, 3] = info[jm5, 2] + info[jm4, 2] + info[jm3, 2] + info[jm2, 2] + new_length2 + new_length1
                x31, x21, y31, y21 = lx - rx, r3x - rx, ly - ry, r3y - ry
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l2x - rx, r4x - rx, l2y - ry, r4y - ry
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm1, 1] = info[jm1, 0] + a1 + a2
                info[jm1, 3] = info[jm4, 2] + info[jm3, 2] + info[jm2, 2] + new_length2 + new_length1 + info[jp1, 2]
                x31, x21, y31, y21 = l2x - nx, r2x - nx, l2y - ny, r2y - ny
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l3x - nx, r3x - nx, l3y - ny, r3y - ny
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[j, 1] = info[j, 0] + a1 + a2
                info[j, 3] = info[jm3, 2] + info[jm2, 2] + new_length2 + new_length1 + info[jp1, 2] + info[jp2, 2]
                x31, x21, y31, y21 = l3x - lx, rx - lx, l3y - ly, ry - ly
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l4x - lx, r2x - lx, l4y - ly, r2y - ly
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jp1, 1] = info[jp1, 0] + a1 + a2
                info[jp1, 3] = info[jm2, 2] + new_length2 + new_length1 + info[jp1, 2] + info[jp2, 2] + info[jp3, 2]
                x31, x21, y31, y21 = l4x - l2x, nx - l2x, l4y - l2y, ny - l2y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l5x - l2x, rx - l2x, l5y - l2y, ry - l2y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jp2, 1] = info[jp2, 0] + a1 + a2
                info[jp2, 3] = new_length2 + new_length1 + info[jp1, 2] + info[jp2, 2] + info[jp3, 2] + info[jp4, 2]
                x31, x21, y31, y21 = l5x - l3x, lx - l3x, l5y - l3y, ly - l3y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l6x - l3x, nx - l3x, l6y - l3y, ny - l3y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jp3, 1] = info[jp3, 0] + a1 + a2
                info[jp3, 3] = new_length1 + info[jp1, 2] + info[jp2, 2] + info[jp3, 2] + info[jp4, 2] + info[jp5, 2]
                quad[i, :] = [o, l, connect_len, r]
            elif update_type == 2:
                connect_count[r] += 1
                connect_count[l2] += 1
                connect[r, l2], connect[l2, r] = True, True
                new_length = ((rx - l2x) ** 2 + (ry - l2y) ** 2) ** 0.5
                total_length -= info[jm1, 2] + info[j, 2] + info[jp1, 2] - new_length
                info[jm1, 2] = new_length
                x31, x21, y31, y21 = l2x - rx, r2x - rx, l2y - ry, r2y - ry
                info[jm1, 0] = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    info[jm1, 0] = _360deg - info[jm1, 0]
                x31, x21, y31, y21 = l3x - l2x, rx - l2x, l3y - l2y, ry - l2y
                info[jp2, 0] = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    info[jp2, 0] = _360deg - info[jp2, 0]
                x31, x21, y31, y21 = rx - r3x, r5x - r3x, ry - r3y, r5y - r3y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l2x - r3x, r6x - r3x, l2y - r3y, r6y - r3y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm3, 1] = info[jm3, 0] + a1 + a2
                info[jm3, 3] = info[jm6, 2] + info[jm5, 2] + info[jm4, 2] + info[jm3, 2] + info[jm2, 2] + new_length
                x31, x21, y31, y21 = l2x - r2x, r4x - r2x, l2y - r2y, r4y - r2y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l3x - r2x, r5x - r2x, l3y - r2y, r5y - r2y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm2, 1] = info[jm2, 0] + a1 + a2
                info[jm2, 3] = info[jm5, 2] + info[jm4, 2] + info[jm3, 2] + info[jm2, 2] + new_length + info[jp2, 2]
                x31, x21, y31, y21 = l3x - rx, r3x - rx, l3y - ry, r3y - ry
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l4x - rx, r4x - rx, l4y - ry, r4y - ry
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm1, 1] = info[jm1, 0] + a1 + a2
                info[jm1, 3] = info[jm4, 2] + info[jm3, 2] + info[jm2, 2] + new_length + info[jp2, 2] + info[jp3, 2]
                x31, x21, y31, y21 = l4x - l2x, r2x - l2x, l4y - l2y, r2y - l2y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l5x - l2x, r3x - l2x, l5y - l2y, r3y - l2y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jp2, 1] = info[jp2, 0] + a1 + a2
                info[jp2, 3] = info[jm3, 2] + info[jm2, 2] + new_length + info[jp2, 2] + info[jp3, 2] + info[jp4, 2]
                x31, x21, y31, y21 = l5x - l3x, rx - l3x, l5y - l3y, ry - l3y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l6x - l3x, r2x - l3x, l6y - l3y, r2y - l3y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jp3, 1] = info[jp3, 0] + a1 + a2
                info[jp3, 3] = info[jm2, 2] + new_length + info[jp2, 2] + info[jp3, 2] + info[jp4, 2] + info[jp5, 2]
                x31, x21, y31, y21 = l6x - l4x, l2x - l4x, l6y - l4y, l2y - l4y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l7x - l4x, rx - l4x, l7y - l4y, ry - l4y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jp4, 1] = info[jp4, 0] + a1 + a2
                info[jp4, 3] = new_length + info[jp2, 2] + info[jp3, 2] + info[jp4, 2] + info[jp5, 2] + info[(j + 6) % lp, 2]
                polygon = delete(polygon, array([j, jp1]))
                info = delete(info, array([j, jp1]), axis = 0)
                quad[i] = o, l, l2, r
                if lp == 6:
                    quad[i + 1] = l2, l3, r2, r
                    break
                lp -= 2
            elif update_type == 1:
                connect_count[r2] += 1
                connect_count[l] += 1
                connect[r2, l], connect[l, r2] = True, True
                new_length = ((r2x - lx) ** 2 + (r2y - ly) ** 2) ** 0.5
                total_length -= info[jm2, 2] + info[jm1, 2] + info[j, 2] - new_length
                info[jm2, 2] = new_length
                x31, x21, y31, y21 = lx - r2x, r3x - r2x, ly - r2y, r3y - r2y
                info[jm2, 0] = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    info[jm2, 0] = _360deg - info[jm2, 0]
                x31, x21, y31, y21 = l2x - lx, r2x - lx, l2y - ly, r2y - ly
                info[jp1, 0] = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    info[jp1, 0] = _360deg - info[jp1, 0]
                x31, x21, y31, y21 = r2x - r4x, r6x - r4x, r2y - r4y, r6y - r4y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = lx - r4x, r7x - r4x, ly - r4y, r7y - r4y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm4, 1] = info[jm4, 0] + a1 + a2
                info[jm4, 3] = info[(j - 7) % lp, 2] + info[jm6, 2] + info[jm5, 2] + info[jm4, 2] + info[jm3, 2] + new_length
                x31, x21, y31, y21 = lx - r3x, r5x - r3x, ly - r3y, r5y - r3y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l2x - r3x, r6x - r3x, l2y - r3y, r6y - r3y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm3, 1] = info[jm3, 0] + a1 + a2
                info[jm3, 3] = info[jm6, 2] + info[jm5, 2] + info[jm4, 2] + info[jm3, 2] + new_length + info[jp1, 2]
                x31, x21, y31, y21 = l2x - r2x, r4x - r2x, l2y - r2y, r4y - r2y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l3x - r2x, r5x - r2x, l3y - r2y, r5y - r2y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm2, 1] = info[jm2, 0] + a1 + a2
                info[jm2, 3] = info[jm5, 2] + info[jm4, 2] + info[jm3, 2] + new_length + info[jp1, 2] + info[jp2, 2]
                x31, x21, y31, y21 = l3x - lx, r3x - lx, l3y - ly, r3y - ly
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l4x - lx, r4x - lx, l4y - ly, r4y - ly
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jp1, 1] = info[jp1, 0] + a1 + a2
                info[jp1, 3] = info[jm4, 2] + info[jm3, 2] + new_length + info[jp1, 2] + info[jp2, 2] + info[jp3, 2]
                x31, x21, y31, y21 = l4x - l2x, r2x - l2x, l4y - l2y, r2y - l2y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l5x - l2x, r3x - l2x, l5y - l2y, r3y - l2y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jp2, 1] = info[jp2, 0] + a1 + a2
                info[jp2, 3] = info[jm3, 2] + new_length + info[jp1, 2] + info[jp2, 2] + info[jp3, 2] + info[jp4, 2]
                x31, x21, y31, y21 = l5x - l3x, lx - l3x, l5y - l3y, ly - l3y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l6x - l3x, r2x - l3x, l6y - l3y, r2y - l3y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jp3, 1] = info[jp3, 0] + a1 + a2
                info[jp3, 3] = new_length + info[jp1, 2] + info[jp2, 2] + info[jp3, 2] + info[jp4, 2] + info[jp5, 2]
                polygon = delete(polygon, array([j, jm1]))
                info = delete(info, array([j, jm1]), axis = 0)
                quad[i] = o, l, r2, r
                if lp == 6:
                    quad[i + 1] = l, l2, r3, r2
                    break
                lp -= 2
            else:
                raw_coordinate = regression2(torch.FloatTensor(dataset1).to(device)).cpu().detach().numpy()
                radius1, radius2, angle1, angle2 = raw_coordinate[0, 1] * reference_length / info[jm1, 2] / 1.2, raw_coordinate[0, 3] * reference_length / info[jm1, 2], raw_coordinate[0, 0] * info[j, 0], raw_coordinate[0, 2] * info[j, 0]
                nx1r, ny1r, nx2r, ny2r = radius1 * cos(angle1), radius1 * sin(angle1), radius2 * cos(angle2), radius2 * sin(angle2)
                connect_count[r] += 1
                connect_count[o] += 1
                connect_len += 1
                connect_count[connect_len], connect[o, connect_len], connect[connect_len, o], connect[connect_len, connect_len + 1], connect[connect_len + 1, connect_len], connect[connect_len + 1, r], connect[r, connect_len + 1] = 2, True, True, True, True, True, True
                connect_len += 1
                connect_count[connect_len], polygon = 2, concatenate((polygon[:j], array([connect_len, connect_len - 1]), polygon[j:]))
                sc2 = s ** 2 + c ** 2
                nx1, ny1, nx2, ny2 = ox + (c * nx1r - s * ny1r) / sc2, oy + (s * nx1r + c * ny1r) / sc2, ox + (c * nx2r - s * ny2r) / sc2, oy + (s * nx2r + c * ny2r) / sc2
                coordinate[connect_len - 1: connect_len + 1, :] = [[nx1, ny1], [nx2, ny2]]
                new_length1, new_length2, new_length3 = ((nx1 - ox) ** 2 + (ny1 - oy) ** 2) ** 0.5, ((nx1 - nx2) ** 2 + (ny1 - ny2) ** 2) ** 0.5, ((nx2 - rx) ** 2 + (ny2 - ry) ** 2) ** 0.5
                total_length -= info[jm1, 2] - new_length1 - new_length2 - new_length3
                info[jm1, 2] = new_length3
                x31, x21, y31, y21 = nx2 - rx, r2x - rx, ny2 - ry, r2y - ry
                info[jm1, 0] = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    info[jm1, 0] = _360deg - info[jm1, 0]
                x31, x21, y31, y21 = lx - ox, nx1 - ox, ly - oy, ny1 - oy
                info[j, 0] = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    info[j, 0] = _360deg - info[j, 0]
                x31, x21, y31, y21 = rx - r3x, r5x - r3x, ry - r3y, r5y - r3y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = nx2 - r3x, r6x - r3x, ny2 - r3y, r6y - r3y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm3, 1] = info[jm3, 0] + a1 + a2
                info[jm3, 3] = info[jm6, 2] + info[jm5, 2] + info[jm4, 2] + info[jm3, 2] + info[jm2, 2] + new_length3
                x31, x21, y31, y21 = nx2 - r2x, r4x - r2x, ny2 - r2y, r4y - r2y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = nx1 - r2x, r5x - r2x, ny1 - r2y, r5y - r2y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm2, 1] = info[jm2, 0] + a1 + a2
                info[jm2, 3] = info[jm5, 2] + info[jm4, 2] + info[jm3, 2] + info[jm2, 2] + new_length3 + new_length2
                x31, x21, y31, y21 = nx1 - rx, r3x - rx, ny1 - ry, r3y - ry
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = ox - rx, r4x - rx, oy - ry, r4y - ry
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jm1, 1] = info[jm1, 0] + a1 + a2
                info[jm1, 3] = info[jm4, 2] + info[jm3, 2] + info[jm2, 2] + new_length3 + new_length2 + new_length1
                x31, x21, y31, y21 = l2x - ox, nx2 - ox, l2y - oy, ny2 - oy
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l3x - ox, rx - ox, l3y - oy, ry - oy
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[j, 1] = info[j, 0] + a1 + a2
                info[j, 3] = new_length3 + new_length2 + new_length1 + info[j, 2] + info[jp1, 2] + info[jp2, 2]
                x31, x21, y31, y21 = l3x - lx, nx1 - lx, l3y - ly, ny1 - ly
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l4x - lx, nx2 - lx, l4y - ly, ny2 - ly
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jp1, 1] = info[jp1, 0] + a1 + a2
                info[jp1, 3] = new_length2 + new_length1 + info[j, 2] + info[jp1, 2] + info[jp2, 2] + info[jp3, 2]
                x31, x21, y31, y21 = l4x - l2x, ox - l2x, l4y - l2y, oy - l2y
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l5x - l2x, nx1 - l2x, l5y - l2y, ny1 - l2y
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                info[jp2, 1] = info[jp2, 0] + a1 + a2
                info[jp2, 3] = new_length1 + info[j, 2] + info[jp1, 2] + info[jp2, 2] + info[jp3, 2] + info[jp4, 2]
                x31, x21, y31, y21 = nx1 - nx2, rx - nx2, ny1 - ny2, ry - ny2
                ni10 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    ni10 = _360deg - ni10
                x31, x21, y31, y21 = ox - nx2, r2x - nx2, oy - ny2, r2y - ny2
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = lx - nx2, r3x - nx2, ly - ny2, r3y - ny2
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                ni11 = ni10 + a1 + a2
                ni12 = info[jm3, 2] + info[jm2, 2] + new_length3 + new_length2 + new_length1 + info[j, 2]
                x31, x21, y31, y21 = ox - nx1, nx2 - nx1, oy - ny1, ny2 - ny1
                ni20 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    ni20 = _360deg - ni20
                x31, x21, y31, y21 = lx - nx1, rx - nx1, ly - ny1, ry - ny1
                a1 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a1 = _360deg - a1
                x31, x21, y31, y21 = l2x - nx1, r2x - nx1, l2y - ny1, r2y - ny1
                a2 = acos(nmax2(-1, nmin2(1, (x31 * x21 + y31 * y21) / ((x21 ** 2 + y21 ** 2) * (x31 ** 2 + y31 ** 2)) ** 0.5)))
                if x21 * y31 < x31 * y21:
                    a2 = _360deg - a2
                ni21 = ni20 + a1 + a2
                ni22 = info[jm2, 2] + new_length3 + new_length2 + new_length1 + info[j, 2] + info[jp1, 2]
                info = concatenate((info[:j], array([[ni10, ni11, new_length2, ni12], [ni20, ni21, new_length1, ni22]]), info[j:]), axis = 0)
                quad[i] = o, connect_len - 1, connect_len, r
                lp += 2

    return coordinate[:(connect_len + 1), :], quad[:(i + 2), :], connect[:(connect_len + 1), :(connect_len + 1)]

f = open('cmu2.txt')
line = f.readline()
line = line.strip('\n')
line = line.split(' ')
line_read = array(line, dtype = float)
total_length = line_read[0]
initnum = int(line_read[1])
polygon = zeros(10 * initnum, dtype = int)
for i in range(initnum):
    polygon[i] = i
line = f.readline()
coordinate = zeros((max_point, 2))
info = zeros((10 * initnum, 4))
for i in range(initnum):
    line = line.strip("\n")
    line = line.split(" ")
    line_read = array(line, dtype = float)
    coordinate[i, :] = line_read[:2]
    info[i, :] = line_read[2:]
    line = f.readline()
line = f.readline()
original_coordinate = zeros((initnum, 2))
count = 0
while line:
    line = line.strip("\n")
    line = line.split(" ")
    line_read = array(line, dtype = float)
    original_coordinate[count, :] = line_read
    line = f.readline()
    count += 1
f.close()
both_possess = equal(coordinate[:, None, :], original_coordinate).all(-1).any(-1)
moveable_points = where(~both_possess)[0]
repeated_points = []
for i in range(initnum - 1):
    for j in range(i + 1, initnum):
        if coordinate[i, 0] == coordinate[j, 0] and coordinate[i, 1] == coordinate[j, 1]:
            exist = False
            for k in range(len(repeated_points)):
                for l in repeated_points[k]:
                    if l == i:
                        exist = True
                        repeated_points[k].append(j)
                        break
                if exist:
                    break
            if not exist:
                repeated_points.append([i, j])
for i in range(moveable_points.shape[0] - 1, -1, -1):
    found = True
    for j in repeated_points:
        for k in range(1, len(j)):
            if moveable_points[i] == j[k]:
                delete(moveable_points, i)
                found = False
                break
        if not found:
            break
initpoints = '*Nset, nset=_0\n'
initcount = 0
for i in range(initnum):
    exist = False
    for j in repeated_points:
        for k in range(1, len(j)):
            if j[k] == i:
                exist = True
                break
        if exist:
            break
    if not exist:
        initpoints += str(i + 1)
        initcount += 1
        if initcount % 16 == 0:
            initpoints += '\n*Nset, nset=_' + str(initcount // 16) + '\n'
        else:
            initpoints += ', '

t0 = time.time()
coordinate_new, quad, connect = implementation(initnum, coordinate, polygon, info, max_episode, max_point, total_length, iga)
print('mesh time:', time.time() - t0)
for i in range(quad.shape[0]):
    for j in range(4):
        if quad[i, j] < initnum:
            exist = False
            for k in repeated_points:
                for l in range(1, len(k)):
                    if k[l] == quad[i, j]:
                        exist = True
                        quad[i, j] = k[0]
                        break
                if exist:
                    break
for i in repeated_points:
    for j in range(1, len(i)):
        get_connection = where(connect[i[j]])
        connect[i[0], get_connection] = True
        connect[get_connection, i[0]] = True
        connect[get_connection, i[j]] = False
        connect[i[j], get_connection] = False

@jit(nopython=True)
def smooth(coordinate_new, quad, moveable_points, connect, point_num, quad_num, iga):
    _180deg = np.pi
    _360deg = 2 * _180deg
    _90deg = 0.5 * _180deg
    move = 1
    while move > 0:
        move = 0
        for i in range(point_num):
            related_quad = quad[np.where(quad == i)[0]]
            ci = np.where(connect[i])[0]
            if ci.shape[0] == 0 or i < initnum and not i in moveable_points:
                continue
            x, y = mean(coordinate_new[connect[i], 0]), mean(coordinate_new[connect[i], 1])
            sec = False
            for j in ci:
                for k in ci:
                    for l in np.where(connect[k])[0]:
                        if i != l and j != k:
                            x1 = x
                            y1 = y
                            x2 = coordinate_new[j, 0]
                            y2 = coordinate_new[j, 1]
                            x3 = coordinate_new[k, 0]
                            y3 = coordinate_new[k, 1]
                            x4 = coordinate_new[l, 0]
                            y4 = coordinate_new[l, 1]
                            if x1 > x2:
                                x12max = x1
                                x12min = x2
                            else:
                                x12max = x2
                                x12min = x1
                            if y1 > y2:
                                y12max = y1
                                y12min = y2
                            else:
                                y12max = y2
                                y12min = y1
                            if x3 > x4:
                                x34max = x3
                                x34min = x4
                            else:
                                x34max = x4
                                x34min = x3
                            if y3 > y4:
                                y34max = y3
                                y34min = y4
                            else:
                                y34max = y4
                                y34min = y3
                            if x12max >= x34min and x34max >= x12min and y12max >= y34min and y34max >= y12min:
                                if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) * (x2 - x1) * (y4 - y1) - (x4 - x1) * (y2 - y1) <= 0 and (x4 - x3) * (y1 - y3) - (x1 - x3) * (y4 - y3) * (x4 - x3) * (y2 - y3) - (x2 - x3) * (y4 - y3) <= 0:
                                    sec = True
                                    break
                    if sec:
                        break
                if sec:
                    break
            if sec:
                continue
            quad_num = related_quad.shape[0]
            coordinate_in_quads = coordinate_new[related_quad.flatten(), :]
            reward_list1 = zeros((quad_num, 2))
            for j in range(quad_num):
                k = 4 * j
                x1, y1, x2, y2, x3, y3, x4, y4 = coordinate_in_quads[k: (k + 4), :].flatten()
                x21, x41, y21, y41, x23, y23, x43, y43 = x2 - x1, x4 - x1, y2 - y1, y4 - y1, x2 - x3, y2 - y3, x4 - x3, y4 - y3
                a1 = acos(nmax2(-1, nmin2(1, (x21 * x41 + y21 * y41) / ((x41 ** 2 + y41 ** 2) * (x21 ** 2 + y21 ** 2)) ** 0.5)))
                if x41 * y21 < x21 * y41:
                    a1= _360deg - a1
                a2 = acos(nmax2(-1, nmin2(1, (x23 * x21 + y23 * y21) / ((x21 ** 2 + y21 ** 2) * (x23 ** 2 + y23 ** 2)) ** 0.5)))
                if x21 * y23 < x23 * y21:
                    a2 = _360deg - a2
                a3 = acos(nmax2(-1, nmin2(1, (x43 * x23 + y43 * y23) / ((x23 ** 2 + y23 ** 2) * (x43 ** 2 + y43 ** 2)) ** 0.5)))
                if x23 * y43 < x43 * y23:
                    a3 = _360deg - a3
                ra1, ra2 = nmin(array([a1, a2, a3, _360deg - a1 - a2 - a3])), _180deg - nmax(array([a1, a2, a3, _360deg - a1 - a2 - a3]))
                if ra1 <= 0 or ra2 <= 0:
                    reward_list1[j, :] = array([0, 0])
                    continue
                d1, d2, d3, d4 = x21 ** 2 + y21 ** 2, x23 ** 2 + y23 ** 2, x43 ** 2 + y43 ** 2, x41 ** 2 + y41 ** 2
                reward_list1[j, :] = array([ra1 * ra2, nmin(array([d1, d2, d3, d4])) / nmax(array([d1, d2, d3, d4]))])
            reward_list1 = power(reward_list1[:, 0] * _90deg ** -2 * power(reward_list1[:, 1], 0.5), 1 / 3)
            xold, yold = coordinate_new[i, :]
            coordinate_new[i, :] = x, y
            coordinate_in_quads = coordinate_new[related_quad.flatten(), :]
            reward_list2 = zeros((quad_num, 2))
            for j in range(quad_num):
                k = 4 * j
                x1, y1, x2, y2, x3, y3, x4, y4 = coordinate_in_quads[k: (k + 4), :].flatten()
                x21, x41, y21, y41, x23, y23, x43, y43 = x2 - x1, x4 - x1, y2 - y1, y4 - y1, x2 - x3, y2 - y3, x4 - x3, y4 - y3
                a1 = acos(nmax2(-1, nmin2(1, (x21 * x41 + y21 * y41) / ((x41 ** 2 + y41 ** 2) * (x21 ** 2 + y21 ** 2)) ** 0.5)))
                if x41 * y21 < x21 * y41:
                    a1= _360deg - a1
                a2 = acos(nmax2(-1, nmin2(1, (x23 * x21 + y23 * y21) / ((x21 ** 2 + y21 ** 2) * (x23 ** 2 + y23 ** 2)) ** 0.5)))
                if x21 * y23 < x23 * y21:
                    a2 = _360deg - a2
                a3 = acos(nmax2(-1, nmin2(1, (x43 * x23 + y43 * y23) / ((x23 ** 2 + y23 ** 2) * (x43 ** 2 + y43 ** 2)) ** 0.5)))
                if x23 * y43 < x43 * y23:
                    a3 = _360deg - a3
                ra1, ra2 = nmin(array([a1, a2, a3, _360deg - a1 - a2 - a3])), _180deg - nmax(array([a1, a2, a3, _360deg - a1 - a2 - a3]))
                if ra1 <= 0 or ra2 <= 0:
                    reward_list2[j, :] = array([0, 0])
                    continue
                d1, d2, d3, d4 = x21 ** 2 + y21 ** 2, x23 ** 2 + y23 ** 2, x43 ** 2 + y43 ** 2, x41 ** 2 + y41 ** 2
                reward_list2[j, :] = array([ra1 * ra2, nmin(array([d1, d2, d3, d4])) / nmax(array([d1, d2, d3, d4]))])
            reward_list2 = power(reward_list2[:, 0] * _90deg ** -2 * power(reward_list2[:, 1], 0.5), 1 / 3)
            if nmin(reward_list2) <= nmin(reward_list1):#mean
                coordinate_new[i, :] = xold, yold
            else:
                move += 1
    return coordinate_new

@jit(nopython = True)
def get_reward(coordinate_in_quads, quad_num, iga):
    _180deg = np.pi
    _360deg = 2 * _180deg
    _90deg = 0.5 * _180deg
    reward_list = zeros(quad_num)
    for i in range(quad_num):
        j = 4 * i
        ax, ay, bx, by, cx, cy, dx, dy = coordinate_in_quads[j: (j + 4), :].flatten()
        x1, y1, x2, y2, x3, y3 = ax, ay, dx, dy, bx, by
        s1 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        x1, y1, x2, y2, x3, y3 = bx, by, ax, ay, cx, cy
        s2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        x1, y1, x2, y2, x3, y3 = cx, cy, bx, by, dx, dy
        s3 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        x1, y1, x2, y2, x3, y3 = dx, dy, cx, cy, ax, ay
        s4 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        reward_list[i] = nmin(array([s1, s2, s3, s4])) / nmax(array([s1, s2, s3, s4]))
    return reward_list

point_num, quad_num = coordinate_new.shape[0], quad.shape[0]
t0 = time.time()
coordinate_new_smooth = smooth(coordinate_new, quad, moveable_points, connect, point_num, quad_num, iga)
t1 = time.time()
r = get_reward(coordinate_new_smooth[quad.flatten(), :], quad_num, iga)
print('reward calculation time: ', time.time() - t1, 'laplacian smoothing time: ', t1 - t0)
print(quad_num, mean(r), min(r))

f = open('mesh.inp', 'w')
f.write('*Heading\n** Job name: Job-1 Model name: Model-1\n** Generated by: Abaqus/CAE 2022\n*Preprint, echo=NO, model=NO, history=NO, contact=NO\n**\n** PARTS\n**\n*Part, name=Part-1\n*Node\n')
for i in range(point_num):
    f.write(str(i + 1) + ', ' + str(coordinate_new_smooth[i, 0]) + ', ' + str(coordinate_new_smooth[i, 1]) + '\n')
f.write('*Element, type=CPS4R\n')
for i in range(quad_num):
    f.write(str(i + 1) + ', ' + str(quad[i, 3] + 1) + ', ' + str(quad[i, 2] + 1) + ', ' + str(quad[i, 1] + 1) + ', ' + str(quad[i, 0] + 1) + '\n')
f.write(initpoints)
initcount2 = initcount
for i in range(initnum, point_num):
    f.write(str(i + 1))
    initcount2 += 1
    if i == point_num - 1:
        f.write('\n')
    elif initcount2 % 16 == 0:
        f.write('\n*Nset, nset=_' + str(initcount2 // 16) + '\n')
    else:
        f.write(', ')
f.write('*End Part\n**\n**\n** ASSEMBLY\n**\n*Assembly, name=Assembly\n**\n*Instance, name=Part-1-1, part=Part-1\n*End Instance\n**\n*End Assembly\n')
f.close()