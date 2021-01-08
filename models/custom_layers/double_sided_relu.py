import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch import nn


def DReLU(x, bias=0, filters=None, epsilon=8.0/255):

    def bias_calculator(filters, epsilon):
        # breakpoint()
        bias = epsilon * torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    if isinstance(filters, torch.Tensor):
        bias = bias_calculator(filters, epsilon)

    # breakpoint()
    return F.relu(x - bias) - F.relu(-x - bias)


def DTReLU(x, bias=0, filters=None, epsilon=8.0/255):

    def bias_calculator(filters, epsilon):
        # breakpoint()
        bias = epsilon * torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    if isinstance(filters, torch.Tensor):
        bias = bias_calculator(filters, epsilon)
    return F.relu(x - bias) + bias * torch.sign(F.relu(x - bias)) - F.relu(-x - bias) - bias * torch.sign(F.relu(-x - bias))


# def TReLU(x):

#     def bias_calculator(filters, epsilon):
#         # breakpoint()
#         bias = epsilon * torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
#         bias = bias.unsqueeze(dim=2)
#         bias = bias.unsqueeze(dim=3)
#         return bias

#     if isinstance(filters, torch.Tensor):
#         bias = bias_calculator(filters, epsilon)

#     return F.relu(x - bias) + bias * torch.sign(F.relu(x - bias))
# def Quad(x):
#     return x**2


class Quad(nn.Module):
    def __init__(self):
        super(Quad, self).__init__()

    def forward(self, x):
        return F.sigmoid(x+1.)-F.sigmoid(x-1.)


class TReLU(nn.Module):

    def __init__(self, in_channels):
        super(TReLU, self).__init__()
        self.noise_level = 255/255.
        self.bias = Parameter(torch.randn((1, in_channels, 1, 1))/10., requires_grad=True)

    def forward(self, x):
        x = x + torch.rand_like(x, device=x.device) * self.noise_level * 2 - self.noise_level
        return F.relu(x - torch.abs(self.bias)) + torch.abs(self.bias) * torch.sign(F.relu(x - torch.abs(self.bias)))


def test_DReLU():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(-10, 10, 0.1), DReLU(torch.Tensor(np.arange(-10, 10, 0.1)), bias=5))
    plt.savefig("double_sided_relu")


if __name__ == '__main__':
    test_DReLU()
