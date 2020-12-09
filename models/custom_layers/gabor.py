import math
from typing import Any

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn.modules import Conv2d, Module
from torch.nn import functional as F


class GaborConv2d(Module):
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

        # small addition to avoid division by zero
        self.delta = 1e-3

        # freq, theta, sigma are set up according to S. Meshgini,
        # A. Aghagolzadeh and H. Seyedarabi, "Face recognition using
        # Gabor filter bank, kernel principal component analysis
        # and support vector machine"
        self.freq = Parameter(
            (math.pi / 2)
            * math.sqrt(2)
            ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor),
            requires_grad=True,
            )
        self.theta = Parameter(
            (math.pi / 8)
            * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),
            requires_grad=True,
            )
        self.sigma = Parameter(math.pi / self.freq, requires_grad=True)
        self.psi = Parameter(
            math.pi * torch.rand(out_channels, in_channels), requires_grad=True
            )

        self.x0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False
            )
        self.y0 = Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False
            )

        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),
                ]
            )
        self.y = Parameter(self.y)
        self.x = Parameter(self.x)

        self.weight = torch.zeros((out_channels, in_channels//groups,
                                   kernel_size[0], kernel_size[1]))

        if bias:
            self.initialize_bias()
        else:
            self.bias = None

        self.initialize_kernels()
        self.normalize_kernels()

        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("x_grid", self.x)

    def forward(self, input_tensor):
        print(self.freq)
        return F.conv2d(input_tensor, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def initialize_kernels(self):
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                g = torch.exp(
                    -0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2)
                    )
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * math.pi * sigma ** 2)
                self.weight[i, j] = g
        self.weight = Parameter(self.weight, requires_grad=False)

    def initialize_bias(self):
        self.bias = nn.Parameter(torch.zeros((self.out_channels)), requires_grad=True)

    def normalize_kernels(self):
        self.weight = Parameter(self.weight /
                                torch.norm(self.weight.view(self.weight.size(0), -1),
                                           p=1, dim=1, keepdim=True).unsqueeze(1).unsqueeze(1), requires_grad=False)
