import torch
from torch import nn

class HyperbolicParaboloidActivation(nn.Module):
    def __init__(self, dim, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype if dtype else torch.float32}
        super(HyperbolicParaboloidActivation, self).__init__()
        self.additive_sigmoid_bias = nn.Parameter(torch.randn((dim,),  **factory_kwargs))
        

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return (x1**2 - x2**2) * torch.sigmoid(x1+self.additive_sigmoid_bias)