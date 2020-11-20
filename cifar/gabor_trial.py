import torch
import matplotlib.pyplot as plt

from ..models.custom_layers import GaborConv2d, DReLU
from ..read_datasets import imagenette
from .parameters import get_arguments

args = get_arguments()

train_loader, test_loader = imagenette(args)
breakpoint()

num_filters = 96
epsilon = 8.0/255
aa = GaborConv2d(in_channels=1, out_channels=num_filters, kernel_size=(11, 11))
aa.calculate_weights()

# fig, axs = plt.subplots(8, 12)
# for ind, ax in enumerate(axs.flat):
#     ax.imshow(aa.conv_layer.weight.detach().numpy()[ind, 0])
# # plt.show()

# aa.calculate_weights()

# fig, axs = plt.subplots(8, 12)
# for ind, ax in enumerate(axs.flat):
#     ax.imshow(aa.conv_layer.weight.detach().numpy()[ind, 0])
# plt.show()

# plt.plot(np.arange(-10, 10, 0.1), DReLU(torch.Tensor(np.arange(-10, 10, 0.1)), 5))
# plt.show()
inp = torch.randn((12, 1, 32, 32))

out = aa(inp)

bias = epsilon * torch.sum(torch.abs(aa.conv_layer.weight), dim=(1, 2, 3)).unsqueeze(dim=0)
bias = bias.unsqueeze(dim=2)
bias = bias.unsqueeze(dim=3)

DReLU(out, bias)

breakpoint()
