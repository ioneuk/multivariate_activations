import torch
from torch import nn
import torch.nn.functional as F

class Geglu1dLearnable(nn.Module):
    def __init__(self, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype if dtype else torch.float32}
        super(Geglu1dLearnable, self).__init__()
        self.mean = nn.Parameter(torch.zeros((1,),  **factory_kwargs))
        self.sigma = nn.Parameter(torch.ones((1,),  **factory_kwargs))

    def forward(self, x):
        return x * F.gelu((x - self.mean) / self.sigma)