import torch
import torch.nn.functional as F


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


def test_DReLU():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(-10, 10, 0.1), DReLU(torch.Tensor(np.arange(-10, 10, 0.1)), bias=5))
    plt.savefig("double_sided_relu")


if __name__ == '__main__':
    test_DReLU()