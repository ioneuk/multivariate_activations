import torch
from torch import nn

@torch.jit.script
def raf_function(num_p, den_p, x):
    x1, x2 = x.chunk(2, dim=-1)
    num = num_p[0, 0] + num_p[0, 1] * x2 + num_p[0, 2] * (x2 ** 2) + num_p[1, 0] * x1 \
        + num_p[1, 1] * x1 * x2 + num_p[1, 2] * x1 * (x2 ** 2) + num_p[2, 0] * (x1 ** 2) \
        + num_p[2, 1] * (x1 ** 2) * x2 + num_p[2, 2] * (x1 ** 2) * (x2 ** 2)
    den = 1 + torch.abs(den_p[0, 0] + den_p[0, 1] * x2 + den_p[0, 2] * (x2 ** 2) + den_p[1, 0] * x1 \
        + den_p[1, 1] * x1 * x2 + den_p[1, 2] * x1 * (x2 ** 2) + den_p[2, 0] * (x1 ** 2) \
        + den_p[2, 1] * (x1 ** 2) * x2 + den_p[2, 2] * (x1 ** 2) * (x2 ** 2))
    return num/den




class Raf2dSecondDegree(nn.Module):
    def __init__(self, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype if dtype else torch.float32}
        super(Raf2dSecondDegree, self).__init__()
        # those params approximate SwiGLU
        self.numerator_params = nn.Parameter(torch.tensor([[-0.0202,  0.1392, -0.0172], 
                                                           [ 0.0522,  1.1218, -0.0035], 
                                                           [ 0.0505,  0.4210, -0.0150]], **factory_kwargs))
        self.denominator_params = nn.Parameter(torch.tensor([[ 0.2945, -0.7757,  0.4774], 
                                                            [ 1.0278,  0.3790, -0.0199], 
                                                            [ 0.5685,  0.1909, -0.4050]], **factory_kwargs))

    def forward(self, x):
        return raf_function(self.numerator_params, self.denominator_params, x)