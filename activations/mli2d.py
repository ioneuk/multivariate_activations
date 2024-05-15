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

class GatedMLISoftLut2LayerWithLearnableCoords(nn.Module):
    def __init__(self, dim, device=None, dtype=None, residual=True):
        super(GatedMLISoftLut2LayerWithLearnableCoords, self).__init__()
        self.dim = dim
        self.residual = residual
        self.scaling_factor = 8

        lut_width = 2
        lut_volume = 2 ** lut_width
        factory_kwargs = {'device': device, 'dtype': dtype if dtype else torch.float32}
        self.weights = nn.Parameter(torch.rand((dim, lut_volume),  **factory_kwargs))
        self.interp_refs = nn.Parameter(torch.tensor([[-0.2, 0.2],[0.2, 0.2], [-0.2, -0.2], [0.2, -0.2]],  **factory_kwargs))
        self._init_weights()

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

        x0, x1, x2, x3 = self.interp_refs[0,0], self.interp_refs[1,0], self.interp_refs[2,0], self.interp_refs[3,0]
        y0, y1, y2, y3 = self.interp_refs[0,1], self.interp_refs[1,1], self.interp_refs[2,1], self.interp_refs[3,1]
        x0_len = x1-x0
        x1_len = x3-x2
        y0_len = y0-y2
        y1_len = y1-y3
        
        ratiox01 = (x1 - input0) / x0_len
        ratiox23 = (x3 - input0) / x1_len
        ratioy02 = (y0 - input1) / y0_len
        ratioy13 = (y1 - input1) / y1_len

        # Expand weights for batch and sequence dimensions
        z3 = self.weights[:, 3].unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)
        z2 = self.weights[:, 2].unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)
        z1 = self.weights[:, 1].unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)
        z0 = self.weights[:, 0].unsqueeze(0).unsqueeze(0).expand(batch_size, seq_length, -1)

        b0 = ratiox01 * z0 + (1 - ratiox01) * z1
        b1 = ratiox23 * z2 + (1 - ratiox23) * z3
        res1 = ratioy02 * b1 + (1 - ratioy02) * b0
        res2 = ratioy13 * b1 + (1 - ratioy13) * b0
        res = (res1 + res2) / 2

        if self.residual:
            res = torch.sigmoid(x) * (x + self.scaling_factor * res)
        else:
            res = res * torch.sigmoid(x)
        return res

    def _init_weights(self, initializer_range=0.02):
        nn.init.normal_(self.weights, std=initializer_range)
