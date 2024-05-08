import torch
from torch import nn

class MLISoftLut2Layer(nn.Module):
    def __init__(self, dim, device=None, dtype=None, input_multiplication = False):
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




class MLISoftInputLut2Layer(nn.Module):
    def __init__(self, device=None, dtype=None, input_multiplication = False):
        super(MLISoftInputLut2Layer, self).__init__()
        self.input_multiplication = input_multiplication

        self.lut_width = 2
        self.lut_volume = 2 ** self.lut_width
        # factory_kwargs = {'device': device, 'dtype': dtype if dtype else torch.float32}
        # self.weights = nn.Parameter(torch.rand((dim, lut_volume),  **factory_kwargs) * 1.8 - 0.9)

    def forward(self, x):
        # Input x expected to be of shape: batch_size x seq_length x dim
        batch_size, seq_length, hidden = x.shape
        assert hidden % 6 == 0

        gate_dim = hidden // 6
        weights_dim = hidden // 6 * 4
        input_dim = hidden // 6
        passthrough, weights, inputs = x.split([gate_dim, weights_dim, input_dim], dim=-1)
        weights = weights.reshape((batch_size, seq_length, -1, self.lut_volume))

        # Generate indices
        idx0 = torch.arange(input_dim, device=x.device) % input_dim
        idx1 = (idx0 + 1) % input_dim

        # Expand indices for batch and sequence dimensions
        idx0 = idx0.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)
        idx1 = idx1.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)

        # Gather inputs using the generated indices
        input1 = torch.gather(inputs, 2, idx1)
        input0 = torch.gather(inputs, 2, idx0)

        ratio1 = (1 - input1) / 2
        ratio0 = (1 - input0) / 2

        # Expand weights for batch and sequence dimensions
        z3 = weights[:,:, :, 3].expand(batch_size, seq_length, -1)
        z2 = weights[:,:,:, 2].expand(batch_size, seq_length, -1)
        z1 = weights[:,:,:, 1].expand(batch_size, seq_length, -1)
        z0 = weights[:,:,:, 0].expand(batch_size, seq_length, -1)

        b1 = ratio1 * z2 + (1 - ratio1) * z3
        b0 = ratio1 * z0 + (1 - ratio1) * z1
        res = ratio0 * b0 + (1 - ratio0) * b1

        # if self.input_multiplication:
        #     res = res * x
        return torch.cat([passthrough, res], dim=-1)


class GatedMLISoftLut2Layer(nn.Module):
    def __init__(self, dim, device=None, dtype=None, residual=True):
        super(GatedMLISoftLut2Layer, self).__init__()
        self.dim = dim
        self.residual = residual

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

        if self.residual:
            res = torch.sigmoid(x) * (x + res)
        else:
            res = res * torch.sigmoid(x)
        return res
