import torch
import torch.nn.functional as F


def TQuantization(x, bias=0, filters=None, epsilon=8.0/255):

    def bias_calculator(filters, epsilon):
        bias = epsilon * torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    if isinstance(filters, torch.Tensor):
        bias = bias_calculator(filters, epsilon)

    return 0.5 * (torch.sign(x - bias) + torch.sign(x + bias))


class TQuantization_BPDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias=0, filters=None, epsilon=8.0/255):

        return TQuantization(x, bias, filters, epsilon)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None


def TSQuantization(x, bias=0, filters=None, epsilon=8.0/255, steepness=100):

    def bias_calculator(filters, epsilon):
        bias = epsilon * torch.sum(torch.abs(filters), dim=(1, 2, 3)).unsqueeze(dim=0)
        bias = bias.unsqueeze(dim=2)
        bias = bias.unsqueeze(dim=3)
        return bias

    if isinstance(filters, torch.Tensor):
        bias = bias_calculator(filters, epsilon)

    # breakpoint()

    return 0.5 * (torch.tanh(steepness*(x - bias)) + torch.tanh(steepness*(x + bias)))


class TSQuantization_BPDA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bias=0, filters=None, epsilon=8.0/255, steepness=100):
        return TSQuantization(x, bias, filters, epsilon, steepness)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


# class activation_quantization_BPDA_smooth_step(torch.autograd.Function):
#     steepness = 0.0

#     def __init__(self, steepness):
#         super(activation_quantization_BPDA_smooth_step, self).__init__()
#         activation_quantization_BPDA_smooth_step.steepness = steepness

#     @staticmethod
#     def forward(ctx, x, l1_norms, jump):
#         # x.shape: batchsize,nb_atoms,L,L

#         x = x / l1_norms.view(1, -1, 1, 1)

#         ctx.save_for_backward(x, jump)

#         x = 0.5 * (torch.sign(x - jump) + torch.sign(x + jump))

#         x = x * l1_norms.view(1, -1, 1, 1)

#         return x

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         Use this if you want an approximation of the activation quantization
#         function in the backward pass. Uses the derivative of
#         0.5*(tanh(bpda_steepness*(x-jump))+tanh(bpda_steepness*(x+jump)))
#         """
#         x, jump = ctx.saved_tensors
#         grad_input = None
#         steepness = activation_quantization_BPDA_smooth_step.steepness

#         def sech(x):
#             return 1 / torch.cosh(x)

#         del_out_over_del_in = 0.5 * steepness * (
#             sech(steepness * (x - jump)) ** 2
#             + sech(steepness * (x + jump)) ** 2
#             )

#         grad_input = del_out_over_del_in * grad_output

#         return grad_input, None, None


# class activation_quantization_BPDA_identity(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, l1_norms, jump):
#         # x.shape: batchsize,nb_atoms,L,L

#         result = x / l1_norms.view(1, -1, 1, 1)

#         result = 0.5 * (torch.sign(result - jump) + torch.sign(result + jump))

#         result = result * l1_norms.view(1, -1, 1, 1)

#         return result

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None, None


def test_TQuantization():
    import numpy as np
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.arange(-10, 10, 0.1), TSQuantization(torch.Tensor(np.arange(-10, 10, 0.1)), bias=5))
    plt.savefig("TSQuantization")


if __name__ == '__main__':
    test_TQuantization()
