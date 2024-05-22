import torch
from torch import nn

# WIP

def _get_xps(z, len_numerator, len_denominator):
    xps = list()
    xps.append(z)
    for _ in range(max(len_numerator, len_denominator) - 2):
        xps.append(xps[-1].mul(z))
    xps.insert(0, torch.ones_like(z))
    return torch.stack(xps, 1)


def Rational_PYTORCH_A_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               1 + | b_1 * X | + | b_2 * X^2| + ... + | b_m * X ^m|

    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    xps = _get_xps(z, len_num, len_deno)
    numerator = xps.mul(weight_numerator).sum(1)
    expanded_dw = torch.cat([torch.tensor([1.]), weight_denominator, \
                             torch.zeros(len_num - len_deno - 1)])
    denominator = xps.mul(expanded_dw).abs().sum(1)
    return numerator.div(denominator).view(x.shape)


def Rational_PYTORCH_B_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               1 + |b_1 * X + b_1 * X^2 + ... + b_m * X^m|
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    xps = _get_xps(z, len_num, len_deno)
    numerator = xps.mul(weight_numerator).sum(1)
    denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1).abs()
    return numerator.div(1 + denominator).view(x.shape)


def Rational_PYTORCH_C_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               eps + |b_0 + b1 * X + b_2 * X^2 + ... + b_m*X^m|
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    xps = _get_xps(z, len_num, len_deno)
    numerator = xps.mul(weight_numerator).sum(1)
    denominator = xps[:, :len_deno].mul(weight_denominator).sum(1).abs()
    return numerator.div(0.1 + denominator).view(x.shape)


def Rational_PYTORCH_D_F(x, weight_numerator, weight_denominator, training, random_deviation=0.1):
    # P(X)/Q(X) = noised(a_0) + noised(a_1) * X +noised(a_2) * X^2 + ... + noised(a_n) * X^n /
    #     #                1 + |noised(b_1) * X + noised(b_2) * X^2 + ... + noised(b_m)*X^m|
    #     # Noised parameters have uniform noise to be in range [(1-random_deviation)*parameter,(1+random_deviation)*parameter].
    if not training:
        # do not add noise
        return Rational_PYTORCH_B_F(x, weight_numerator, weight_denominator, training)
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    xps = _get_xps(z, len_num, len_deno)
    numerator = xps.mul(weight_numerator.mul(
        torch.FloatTensor(len_num).uniform_(1-random_deviation,
                                            1+random_deviation))
                       ).sum(1)
    denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1).abs()
    return numerator.div(1 + denominator).view(x.shape)

def Rational_NONSAFE_F(x, weight_numerator, weight_denominator, training):
    # P(X) / Q(X) = a_0 + a_1 * X + ... + a_n * X^n /
    #               1 + b_1 * X + b_1 * X^2 + ... + b_m * X^m
    z = x.view(-1)
    len_num, len_deno = len(weight_numerator), len(weight_denominator)
    # xps = torch.vander(z, max(len_num, len_deno), increasing=True)
    xps = _get_xps(z, len_num, len_deno).to(weight_numerator.device)
    numerator = xps.mul(weight_numerator).sum(1)
    denominator = xps[:, 1:len_deno+1].mul(weight_denominator).sum(1)
    return numerator.div(1 + denominator).view(x.shape)


class Rational(nn.Module):
    """
    Rational activation function inherited from ``torch.nn.Module``.

    Arguments:
            approx_func (str):
                The name of the approximated function for initialisation. \
                The different initialable functions are available in \
                `rational.rationals_config.json`. \n
                Default ``leaky_relu``.
            degrees (tuple of int):
                The degrees of the numerator (P) and denominator (Q).\n
                Default ``(5, 4)``
            cuda (bool):
                Use GPU CUDA version. \n
                If ``None``, use cuda if available on the machine\n
                Default ``None``
            version (str):
                Version of Rational to use. Rational(x) = P(x)/Q(x)\n
                `A`: Q(x) = 1 + \|b_1.x\| + \|b_2.x\| + ... + \|b_n.x\|\n
                `B`: Q(x) = 1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                `C`: Q(x) = 0.1 + \|b_1.x + b_2.x + ... + b_n.x\|\n
                `D`: like `B` with noise\n
                Default ``A``
            trainable (bool):
                If the weights are trainable, i.e, if they are updated during \
                backward pass\n
                Default ``True``
    Returns:
        Module: Rational module
    """

    def __init__(self, approx_func="leaky_relu", degrees=(5, 4),
                 version="A", trainable=True, train_numerator=True,
                 train_denominator=True, name=None, device=None):

        if name is None:
            name = approx_func
        super().__init__(name)


        w_numerator, w_denominator = get_parameters(version, degrees,
                                                    approx_func)

        self.numerator = nn.Parameter(torch.FloatTensor(w_numerator).to(device),
                                      requires_grad=trainable and train_numerator)
        self.denominator = nn.Parameter(torch.FloatTensor(w_denominator).to(device),
                                        requires_grad=trainable and train_denominator)
        self.register_parameter("numerator", self.numerator)
        self.register_parameter("denominator", self.denominator)
        self.device = device
        self.degrees = degrees
        self.version = version
        self.training = trainable

        self.init_approximation = approx_func
        self._saving_input = False

        
        if version == "A":
            rational_func = Rational_PYTORCH_A_F
        elif version == "B":
            rational_func = Rational_PYTORCH_B_F
        elif version == "C":
            rational_func = Rational_PYTORCH_C_F
        elif version == "D":
            rational_func = Rational_PYTORCH_D_F
        elif version == "N":
            rational_func = Rational_NONSAFE_F
        else:
            raise NotImplementedError(f"version {version} not implemented")

        self.activation_function = rational_func

    def forward(self, x):
        return self.activation_function(x, self.numerator, self.denominator,
                                        self.training)