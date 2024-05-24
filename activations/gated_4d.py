import torch
from torch import nn
import torch.nn.functional as F

class Gated4d(nn.Module):
    def __init__(self):
        super(Gated4d, self).__init__()

    def forward(self, x):
        unfolded = x.unfold(dimension=-1, size=4, step=2)
        res = (unfolded[:,:,:,0] + unfolded[:,:,:,1]) * (unfolded[:,:,:,2] - unfolded[:,:,:,3])
        return res

class Gated4dV2(nn.Module):
    def __init__(self):
        super(Gated4dV2, self).__init__()

    def forward(self, x):
        x1, x2, x3, x4 = x.chunk(4, dim=-1)
        res = x1 * torch.sigmoid(x2) * (x3+x4)
        return res
