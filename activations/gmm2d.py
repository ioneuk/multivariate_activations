import torch
import torch.nn as nn
from torch import Tensor

from activations.gmm_triton import GMM2DTriton



def _eval_2d_gaussian_precomputed(mean: Tensor, inv_var_covar: Tensor, det: Tensor, x: Tensor):
    term1 = 2 * torch.pi * torch.sqrt(det)
    diff = (x - mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)).unsqueeze(-2)
    term2 = -0.5 * torch.matmul(torch.matmul(diff, inv_var_covar), diff.transpose(-1, -2))
    pdf = torch.exp(term2) / term1
    return pdf.squeeze()


@torch.compile
def gmm2d_precomputed(x: Tensor, means: Tensor, inv_var_covar: Tensor, det: Tensor, weights: Tensor):
    x = x.view(x.shape[0], x.shape[1], -1, 2)
    n_components, _ = means.shape
    batch_size, seq_len, dim, _ = x.shape
    # result = torch.zeros(batch_size, seq_len, dim, 2, dtype=weights.dtype, device=weights.device)
    result = torch.zeros_like(x).reshape(batch_size, seq_len, dim, 2)
    for i in range(n_components):
        _m = means[i]
        _covar = inv_var_covar[i]
        _det = det[i]
        result += weights[:, :, :, i] * _eval_2d_gaussian_precomputed(_m, _covar, _det, x).unsqueeze(-1)
    return result.reshape(batch_size, seq_len, -1)


class GMMActivation2D(nn.Module):
    def __init__(self, dim: int, device=None, dtype=None, use_triton=True):
        factory_kwargs = {'device': device, 'dtype': dtype if dtype else torch.float32}
        super(GMMActivation2D, self).__init__()
        self.dim = dim
        self.use_triton = use_triton
        self.modes = nn.Parameter(torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], **factory_kwargs), requires_grad=False)
        self.weights = nn.Parameter(torch.empty((1, 1, dim // 2, 4, 2), **factory_kwargs).normal_())
        self.var_covar = nn.Parameter(torch.tensor([[[2.2458, -0.0288], [-0.0288, 2.2458]],
                                                    [[2.2397, 0.0118], [0.0118, 2.2397]],
                                                    [[2.2436, 0.0172], [0.0172, 2.2436]],
                                                    [[2.2394, -0.0123], [-0.0123, 2.2394]]], **factory_kwargs), requires_grad=False)
        self.inv_var_covar = nn.Parameter(torch.linalg.inv(self.var_covar), requires_grad=False)
        self.det_var_covar = nn.Parameter(torch.tensor([torch.det(self.var_covar[i]) for i in range(4)], **factory_kwargs), requires_grad=False)

    def forward(self, x):
        if self.use_triton:
            return GMM2DTriton.apply(x, self.weights)
        else:
            return gmm2d_precomputed(x, self.modes, self.inv_var_covar, self.det_var_covar, self.weights)

class GMMActivation2DFullyLearnable(nn.Module):
    def __init__(self, dim, num_components=4, device=None, dtype=None, use_triton=True):
        super(GMMActivation2DFullyLearnable, self).__init__()
        self.dim = dim
        self.num_components = num_components
        self.weights = nn.Parameter(torch.randn(1, 1, dim // 2, 4, 2))
        self.means = nn.Parameter(torch.randn(num_components, 2))
        
        rand_mat = torch.randn((num_components, 2, 2))
        var_covar = nn.Parameter(torch.bmm(rand_mat, rand_mat.transpose(1, 2)))
        self.inv_var_covar = nn.Parameter(torch.linalg.inv(var_covar), requires_grad=False)


    def forward(self, x):
        return x * gmm2d(x, self.means, self.inv_var_covar, self.weights)


def _eval_2d_gaussian(mean: Tensor, inv_var_covar: Tensor, x: Tensor):
    diff = (x - mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)).unsqueeze(-2)
    term2 = -0.5 * torch.matmul(torch.matmul(diff, inv_var_covar), diff.transpose(-1, -2))
    pdf = torch.exp(term2)
    return pdf.squeeze()


@torch.compile
def gmm2d(x: Tensor, means: Tensor, inv_var_covar: Tensor, weights: Tensor):
    x = x.view(x.shape[0], x.shape[1], -1, 2)
    n_components, _ = means.shape
    batch_size, seq_len, dim, _ = x.shape
    result = torch.zeros_like(x).reshape(batch_size, seq_len, dim, 2)
    for i in range(n_components):
        _m = means[i]
        _covar = inv_var_covar[i]
        result += weights[:, :, :, i] * _eval_2d_gaussian(_m, _covar, x).unsqueeze(-1)
    
    return result.reshape(batch_size, seq_len, -1)
