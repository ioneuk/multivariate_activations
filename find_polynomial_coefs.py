import torch

from torch import Tensor, nn
from tqdm import tqdm
from transformers import AdamW


def swiglu(x1: Tensor, x2: Tensor):
    return x1 * torch.sigmoid(x1) * x2

def swiglu_batched(x: Tensor):
    return x[:,0] * torch.sigmoid(x[:,0]) * x[:,1]


def polymomial_degree_2(coefs: Tensor, x1, x2):
    assert coefs.shape == (3, 3)
    return coefs[0, 0] + coefs[0, 1] * x2 + coefs[0, 2] * (x2 ** 2) + coefs[1, 0] * x1 + coefs[1, 1] * x1 * x2 \
        + coefs[1, 2] * x1 * (x2 ** 2) + coefs[2, 0] * (x1 ** 2) + coefs[2, 1] * (x1 ** 2) * x2 \
        + coefs[2, 2] * (x1 ** 2) * (x2 ** 2)

def polymomial_degree_2_batched(coefs: Tensor, x):
    assert coefs.shape == (3, 3)

    return (coefs[0, 0] + coefs[0, 1] * x[:,1] + coefs[0, 2] * (x[:,1] ** 2) + coefs[1, 0] * x[:,0] \
        + coefs[1, 1] * x[:,0] * x[:,1] + coefs[1, 2] * x[:,0] * (x[:,1] ** 2) + coefs[2, 0] * (x[:,0] ** 2) \
        + coefs[2, 1] * (x[:,0] ** 2) * x[:,1] + coefs[2, 2] * (x[:,0] ** 2) * (x[:,1] ** 2))

def raf_2_batched(coefs_n: Tensor, coefs_d: Tensor, x):
    assert coefs_n.shape == (3, 3)
    assert coefs_d.shape == (3, 3)
    num = coefs_n[0, 0] + coefs_n[0, 1] * x[:,1] + coefs_n[0, 2] * (x[:,1] ** 2) + coefs_n[1, 0] * x[:,0] \
        + coefs_n[1, 1] * x[:,0] * x[:,1] + coefs_n[1, 2] * x[:,0] * (x[:,1] ** 2) + coefs_n[2, 0] * (x[:,0] ** 2) \
        + coefs_n[2, 1] * (x[:,0] ** 2) * x[:,1] + coefs_n[2, 2] * (x[:,0] ** 2) * (x[:,1] ** 2)
    den = 1 + torch.abs(coefs_d[0, 0] + coefs_d[0, 1] * x[:,1] + coefs_d[0, 2] * (x[:,1] ** 2) + coefs_d[1, 0] * x[:,0] \
        + coefs_d[1, 1] * x[:,0] * x[:,1] + coefs_d[1, 2] * x[:,0] * (x[:,1] ** 2) + coefs_d[2, 0] * (x[:,0] ** 2) \
        + coefs_d[2, 1] * (x[:,0] ** 2) * x[:,1] + coefs_d[2, 2] * (x[:,0] ** 2) * (x[:,1] ** 2))
    return num/den

def raf_function_3_deg(num_p, den_p, x):
    assert num_p.shape == (4, 4)
    assert den_p.shape == (4, 4)
    x1, x2 = x.chunk(2, dim=-1)
    num = num_p[0, 0] + num_p[0, 1] * x2 + num_p[0, 2] * (x2 ** 2) + num_p[0, 3] * (x2**3) \
          + num_p[1, 0] * x1 + num_p[1, 1] * x1 * x2 + num_p[1, 2] * x1 * (x2 ** 2) + + num_p[1, 3] * x1 * (x2**3) \
          + num_p[2, 0] * (x1 ** 2) + num_p[2, 1] * (x1 ** 2) * x2 + num_p[2, 2] * (x1 ** 2) * (x2 ** 2) + num_p[2, 3] * (x1**2) * (x2**3) \
          + num_p[3, 0] * (x1 ** 3) + num_p[3, 1] * (x1 ** 3) * x2 + num_p[3, 2] * (x1 ** 3) * (x2 ** 2) + num_p[3, 3] * (x1**3) * (x2**3)
    den = 1 + torch.abs(den_p[0, 0] + den_p[0, 1] * x2 + den_p[0, 2] * (x2 ** 2) + den_p[0, 3] * (x2 ** 3) \
                        + den_p[1, 0] * x1 + den_p[1, 1] * x1 * x2 + den_p[1, 2] * x1 * (x2 ** 2) + den_p[1, 3] * x1 * (x2 ** 3) \
                        + den_p[2, 0] * (x1 ** 2) + den_p[2, 1] * (x1 ** 2) * x2 + den_p[2, 2] * (x1 ** 2) * (x2 ** 2) + den_p[2, 3] * (x1**2) * (x2**3) \
                        + den_p[3, 0] * (x1 ** 3) + den_p[3, 1] * (x1 ** 3) * x2 + den_p[3, 2] * (x1 ** 3) * (x2 ** 2) + den_p[3, 3] * (x1**3) * (x2**3))
    return num/den

def raf_1_batched(coefs_n: Tensor, coefs_d: Tensor, x):
    assert coefs_n.shape == (2, 2)
    assert coefs_d.shape == (2, 2)
    num = coefs_n[0, 0] + coefs_n[0, 1] * x[:,1] + coefs_n[1, 0] * x[:,0] \
        + coefs_n[1, 1] * x[:,0] * x[:,1]
    den = 1 + torch.abs(coefs_d[0, 0] + coefs_d[0, 1] * x[:,1] + coefs_d[1, 0] * x[:,0] \
        + coefs_d[1, 1] * x[:,0] * x[:,1])
    return num/den


if __name__ == '__main__':
    polynomial_degree = 3
    n_iterations = 10000
    batch_size = 100
    print_loss_every_n_iters = 200
    coefs_n = torch.randn((polynomial_degree + 1, polynomial_degree + 1), requires_grad=True, device='cuda')
    coefs_d = torch.randn((polynomial_degree + 1, polynomial_degree + 1), requires_grad=True, device='cuda')
    optimizer = AdamW(params=[coefs_n, coefs_d], weight_decay=0.1)
    loss_fn = nn.MSELoss()
    for i in tqdm(range(n_iterations)):
        optimizer.zero_grad()
        xy = torch.randn((100, 2), device='cuda')
        gt = swiglu_batched(xy)
        pred = raf_function_3_deg(coefs_n, coefs_d, xy)
        loss = loss_fn(pred, gt)
        loss.backward()
        optimizer.step()
        if i % print_loss_every_n_iters == 0:
            print(f'Loss: {loss}')

    print('Numerator:')
    print(coefs_n)
    print('Denominator:')
    print(coefs_d)
