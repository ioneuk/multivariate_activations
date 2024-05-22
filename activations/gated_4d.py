from torch import nn
import torch.nn.functional as F

class Gated4d(nn.Module):
    def __init__(self):
        super(Gated4d, self).__init__()

    def forward(self, x):
        unfolded = x.unfold(dimension=-1, size=4, step=2)
        res = unfolded[:,:,:,0] * unfolded[:,:,:,1] + unfolded[:,:,:,2] * unfolded[:,:,:,3]
        return res