import torch
from torch import nn

class MLISoftLut2Layer(nn.Module):
    def __init__(self, dim, device=None, dtype=None, input_multiplication = True):
        super(MLISoftLut2Layer, self).__init__()
        self.dim = dim
        self.input_multiplication = input_multiplication

        lut_width = 2
        lut_volume = 2 ** lut_width
        factory_kwargs = {'device': device, 'dtype': dtype if dtype else torch.float32}
        self.weights = nn.Parameter(torch.rand((dim, lut_volume),  **factory_kwargs) * 1.8 - 0.9)

    def forward(self, x):
        # Input x expected to be of shape: batch_size x seq_length x dim
        batch_size, seq_length, _ = x.shape

        # Generate indices
        idx0 = torch.arange(self.dim, device=x.device) % self.dim
        idx1 = (idx0 + 1) % self.dim

        # Expand indices for batch and sequence dimensions
        idx0 = idx0.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)
        idx1 = idx1.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)

        # Gather inputs using the generated indices
        input1 = torch.gather(x, 2, idx1)
        input0 = torch.gather(x, 2, idx0)

        ratio1 = (1 - input1) / 2
        ratio0 = (1 - input0) / 2

        # Expand weights for batch and sequence dimensions
        z3 = self.weights[:, 3].unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)
        z2 = self.weights[:, 2].unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)
        z1 = self.weights[:, 1].unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)
        z0 = self.weights[:, 0].unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)

        b1 = ratio1 * z2 + (1 - ratio1) * z3
        b0 = ratio1 * z0 + (1 - ratio1) * z1
        res = ratio0 * b0 + (1 - ratio0) * b1

        if self.input_multiplication:
            res = res * x
        return res
