import torch
from torch import nn

@torch.jit.script
def raf_function_1_deg(num_p, den_p, x):
    x1, x2 = x.chunk(2, dim=-1)
    num = num_p[0, 0] + num_p[0, 1] * x2 + num_p[1, 0] * x1 + num_p[1, 1] * x1 * x2
    den = 1 + torch.abs(den_p[0, 0] + den_p[0, 1] * x2 + den_p[1, 0] * x1 + den_p[1, 1] * x1 * x2)
    return num / den

@torch.jit.script
def raf_function_2_deg(num_p, den_p, x):
    x1, x2 = x.chunk(2, dim=-1)
    num = num_p[0, 0] + num_p[0, 1] * x2 + num_p[0, 2] * (x2 ** 2) + num_p[1, 0] * x1 \
          + num_p[1, 1] * x1 * x2 + num_p[1, 2] * x1 * (x2 ** 2) + num_p[2, 0] * (x1 ** 2) \
          + num_p[2, 1] * (x1 ** 2) * x2 + num_p[2, 2] * (x1 ** 2) * (x2 ** 2)
    den = 1 + torch.abs(den_p[0, 0] + den_p[0, 1] * x2 + den_p[0, 2] * (x2 ** 2) + den_p[1, 0] * x1 \
                        + den_p[1, 1] * x1 * x2 + den_p[1, 2] * x1 * (x2 ** 2) + den_p[2, 0] * (x1 ** 2) \
                        + den_p[2, 1] * (x1 ** 2) * x2 + den_p[2, 2] * (x1 ** 2) * (x2 ** 2))
    return num / den

@torch.jit.script
def raf_function_3_deg(num_p, den_p, x):
    x1, x2 = x.chunk(2, dim=-1)
    num = num_p[0, 0] + num_p[0, 1] * x2 + num_p[0, 2] * (x2 ** 2) + num_p[0, 3] * (x2 ** 3) \
          + num_p[1, 0] * x1 + num_p[1, 1] * x1 * x2 + num_p[1, 2] * x1 * (x2 ** 2) + num_p[1, 3] * x1 * (x2 ** 3) \
          + num_p[2, 0] * (x1 ** 2) + num_p[2, 1] * (x1 ** 2) * x2 + num_p[2, 2] * (x1 ** 2) * (x2 ** 2) + num_p[2, 3] * (x1 ** 2) * (x2 ** 3) \
          + num_p[3, 0] * (x1 ** 3) + num_p[3, 1] * (x1 ** 3) * x2 + num_p[3, 2] * (x1 ** 3) * (x2 ** 2) + num_p[3, 3] * (x1 ** 3) * (x2 ** 3)
    den = 1 + torch.abs(den_p[0, 0] + den_p[0, 1] * x2 + den_p[0, 2] * (x2 ** 2) + den_p[0, 3] * (x2 ** 3) \
                        + den_p[1, 0] * x1 + den_p[1, 1] * x1 * x2 + den_p[1, 2] * x1 * (x2 ** 2) + den_p[1, 3] * x1 * (x2 ** 3) \
                        + den_p[2, 0] * (x1 ** 2) + den_p[2, 1] * (x1 ** 2) * x2 + den_p[2, 2] * (x1 ** 2) * (x2 ** 2) + den_p[2, 3] * (x1 ** 2) * (x2 ** 3) \
                        + den_p[3, 0] * (x1 ** 3) + den_p[3, 1] * (x1 ** 3) * x2 + den_p[3, 2] * (x1 ** 3) * (x2 ** 2) + den_p[3, 3] * (x1 ** 3) * (x2 ** 3))
    return num / den

class Raf2dFirstDegree(nn.Module):
    def __init__(self, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype if dtype else torch.float32}
        super(Raf2dFirstDegree, self).__init__()
        # those params approximate SwiGLU
        self.numerator_params = nn.Parameter(torch.tensor([[-0.0012, 0.1872],
                                                           [-0.0043, 0.9203]], **factory_kwargs))
        self.denominator_params = nn.Parameter(torch.tensor([[1.1464e+00, -3.5197e-04],
                                                             [-5.0235e-01, -2.1131e-03]], **factory_kwargs))

    def forward(self, x):
        res = raf_function_1_deg(self.numerator_params, self.denominator_params, x)

        return res

class Raf2dSecondDegree(nn.Module):
    def __init__(self, device=None, dtype=None, swish_mult=True):
        factory_kwargs = {'device': device, 'dtype': dtype if dtype else torch.float32}
        super(Raf2dSecondDegree, self).__init__()
        # those params approximate SwiGLU
        self.numerator_params = nn.Parameter(torch.tensor([[-0.0202, 0.1392, -0.0172],
                                                           [0.0522, 1.1218, -0.0035],
                                                           [0.0505, 0.4210, -0.0150]], **factory_kwargs))
        self.denominator_params = nn.Parameter(torch.tensor([[0.2945, -0.7757, 0.4774],
                                                             [1.0278, 0.3790, -0.0199],
                                                             [0.5685, 0.1909, -0.4050]], **factory_kwargs))
        self.swish_mult = swish_mult

    def forward(self, x):
        res = raf_function_2_deg(self.numerator_params, self.denominator_params, x)
        if self.swish_mult:
            gate, _ = x.chunk(2, dim=-1)
            res = res * gate * torch.sigmoid(gate)
        return res


class Raf2dThirdDegree(nn.Module):
    def __init__(self, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype if dtype else torch.float32}
        super(Raf2dThirdDegree, self).__init__()
        # those params approximate SwiGLU
        self.numerator_params = nn.Parameter(torch.tensor([[-0.0070, 0.0095, 0.0106, -0.0038],
                                                           [-0.0176, 0.1629, 0.0218, -0.0750],
                                                           [0.0270, -0.0143, -0.0176, 0.0087],
                                                           [0.0219, -0.1454, -0.0147, 0.0820]], **factory_kwargs))
        self.denominator_params = nn.Parameter(torch.tensor([[-0.5213, -0.3284, -0.2400, 0.1468],
                                                             [0.0445, 0.2835, 0.2114, -0.1922],
                                                             [-0.1562, 0.9840, 0.4270, -0.1576],
                                                             [-0.7061, 0.4117, -0.8121, -0.0229]], **factory_kwargs))

    def forward(self, x):
        res = raf_function_3_deg(self.numerator_params, self.denominator_params, x)
        return res
