import math
from typing import Any

import torch
from torch.nn import functional as F
from torch.nn import Parameter
from torch.nn.modules import Conv2d, Module


class LowPassConv2d(Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
            ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = torch.zeros((out_channels, in_channels//groups,
                                   self.kernel_size[0], self.kernel_size[1]))

        self.initialize_kernels()
        if bias:
            self.initialize_bias()
        else:
            self.bias = None

        self.register_parameter("weight", self.weight)
        # self.register_parameter("bias", self.bias)

    def forward(self, input_tensor):
        return F.conv2d(input_tensor, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def initialize_kernels(self):
        self.weight = torch.ones_like(self.weight)/(self.kernel_size[0]*self.kernel_size[1])
        self.weight = Parameter(self.weight, requires_grad=False)

    def initialize_bias(self):
        self.bias = Parameter(torch.zeros((self.out_channels)), requires_grad=True)

    def normalize_kernels(self):
        self.weight = Parameter(self.weight /
                                torch.norm(self.weight.view(self.weight.size(0), -1),
                                           p=1, dim=1, keepdim=True).unsqueeze(1).unsqueeze(1), requires_grad=False)
