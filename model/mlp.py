from torch import nn
import torch.nn.functional as F

from activations.mli2d import MLISoftInputLut2Layer

class InterpolationMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features_first=None,
        hidden_features_second=None,
        out_features=None,
        activation=MLISoftInputLut2Layer,
        bias1=True,
        bias2=True,
        return_residual=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features_first = hidden_features_first if hidden_features_first is not None else in_features * 4
        hidden_features_second = hidden_features_second if hidden_features_second is not None else hidden_features_first // 5 
        self.return_residual = return_residual
        self.fc1 = nn.Linear(in_features, hidden_features_first, bias=bias1, **factory_kwargs)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_features_second, out_features, bias=bias2, **factory_kwargs)

    def forward(self, x):
        y = self.fc1(x)
        y = self.activation(y)
        y = self.fc2(y)
        return y if not self.return_residual else (y, x)

