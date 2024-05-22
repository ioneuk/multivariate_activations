import torch
from torch import nn
import torch.nn.functional as F

class GatedMish(nn.Module):
    def __init__(self):
        super(GatedMish, self).__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * x2 * (torch.log(1+torch.exp(x1+x2)))