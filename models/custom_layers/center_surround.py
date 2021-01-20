"""
Neural Network models for training and testing implemented in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...utils import l1_normalizer, DifferenceOfGaussian2d, rgb2yrgb, rgb2y
from . import DReLU, DTReLU, TQuantization, TSQuantization


class CenterSurroundModule(nn.Module):

    def __init__(self, beta=1.0, freeze_weights=True):
        super(CenterSurroundModule, self).__init__()

        self.laplacian = l1_normalizer(np.array([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]))
        self.low_pass = l1_normalizer(np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]))
        self.beta = beta

        filters = np.concatenate(
            (self.laplacian, self.low_pass, self.laplacian, self.low_pass, self.laplacian, self.low_pass))

        self.basis_expansion = torch.nn.Conv2d(
            in_channels=3, out_channels=6, stride=1, kernel_size=3, padding=1, groups=3, bias=False)
        self.reconstruction = torch.nn.ConvTranspose2d(
            in_channels=6, out_channels=3, stride=1, kernel_size=3, padding=1, groups=3, bias=False)

        self.basis_expansion.weight.data = torch.Tensor(filters).unsqueeze(1)
        self.reconstruction.weight.data = torch.Tensor(filters).unsqueeze(1)

        # Freezing weights
        self.basis_expansion.weight.requires_grad = not freeze_weights
        self.reconstruction.weight.requires_grad = not freeze_weights

    def forward(self, x):

        o = self.basis_expansion(x)
        o = DTReLU(o, filters=self.basis_expansion.weight, epsilon=self.beta*8.0/255)
        o = self.reconstruction(o)

        return o

    def firstlayer(self, x):
        o = self.basis_expansion(x)
        return o, DTReLU(o, filters=self.basis_expansion.weight, epsilon=self.beta*8.0/255)


class CenterSurroundConv(nn.Module):

    def __init__(self, beta=1.0, freeze_weights=True):
        super(CenterSurroundConv, self).__init__()

        self.laplacian = l1_normalizer(np.array([[[0, -1, 0], [-1, 4, -1], [0, -1, 0]]]))
        self.low_pass = l1_normalizer(np.array([[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]))
        self.beta = beta

        filters = np.concatenate(
            (self.laplacian, self.low_pass, self.laplacian, self.low_pass, self.laplacian, self.low_pass))

        self.basis_expansion = torch.nn.Conv2d(
            in_channels=3, out_channels=6, stride=1, kernel_size=3, padding=1, groups=3, bias=False)
        self.reconstruction = torch.nn.ConvTranspose2d(
            in_channels=6, out_channels=3, stride=1, kernel_size=3, padding=1, bias=True)

        self.basis_expansion.weight.data = torch.Tensor(filters).unsqueeze(1)

        # Freezing weights
        self.basis_expansion.weight.requires_grad = not freeze_weights

    def forward(self, x):

        o = self.basis_expansion(x)
        # o = o + torch.rand_like(o, device=o.device) * 16./255 - 8./255
        o = DTReLU(o, filters=self.basis_expansion.weight, epsilon=self.beta*8.0/255)
        o = self.reconstruction(o)

        return o

    # def firstlayer(self, x):
    #     o = self.basis_expansion(x)
    #     return o, DReLU(o, filters=self.basis_expansion.weight, epsilon=8.0/255)


class DoGLayer(nn.Module):

    def __init__(self, beta=1.0, freeze_weights=True):
        super(DoGLayer, self).__init__()

        self.DoG = DifferenceOfGaussian2d(kernel_size=5, sigma1=0.5, sigma2=0.8, normalize=True)
        self.beta = beta

        filters = np.stack([self.DoG.kernel, self.DoG.kernel, self.DoG.kernel], axis=0)

        self.basis_expansion = torch.nn.Conv2d(
            in_channels=3, out_channels=3, stride=1, kernel_size=5, padding=2, groups=3, bias=False)

        self.basis_expansion.weight.data = torch.Tensor(filters).unsqueeze(1)

        # Freezing weights
        self.basis_expansion.weight.requires_grad = not freeze_weights

    def forward(self, x):

        o = self.basis_expansion(x)
        # o = DTReLU(o, filters=self.basis_expansion.weight, epsilon=self.beta*8.0/255)
        o = TSQuantization(o, filters=self.basis_expansion.weight, epsilon=self.beta*8.0/255)

        return o


class DoGLowpassLayer(nn.Module):

    def __init__(self, beta=1.0, freeze_weights=True):
        super().__init__()

        self.DoG = DifferenceOfGaussian2d(kernel_size=5, sigma1=0.6, sigma2=1.0, normalize=True)
        self.beta = beta

        self.low_pass = l1_normalizer(np.ones((5, 5)))

        lp_filters = np.stack([self.low_pass, self.low_pass, self.low_pass], axis=0)

        self.dog = torch.nn.Conv2d(
            in_channels=1, out_channels=1, stride=1, kernel_size=5, padding=2, groups=1, bias=False)
        self.lp = torch.nn.Conv2d(
            in_channels=3, out_channels=3, stride=1, kernel_size=5, padding=2, groups=3, bias=False)

        self.dog.weight.data = torch.Tensor(self.DoG.kernel).unsqueeze(0).unsqueeze(1)
        self.lp.weight.data = torch.Tensor(lp_filters).unsqueeze(1)

        # Freezing weights
        self.dog.weight.requires_grad = not freeze_weights
        self.lp.weight.requires_grad = not freeze_weights

    def forward(self, x):
        y = rgb2y(x)
        yo = 1. - torch.abs(self.dog(y))
        yo = TSQuantization(yo, filters=self.dog.weight, epsilon=self.beta*8.0/255)
        xo = self.lp(x)
        xo = TSQuantization(xo, filters=self.lp.weight, epsilon=16*self.beta*8.0/255)
        o = yo.expand_as(xo) + xo
        # breakpoint()

        return o


class LowpassLayer(nn.Module):

    def __init__(self, beta=1.0, freeze_weights=True):
        super().__init__()

        self.beta = beta

        self.low_pass = l1_normalizer(np.ones((5, 5)))

        lp_filters = np.stack([self.low_pass, self.low_pass, self.low_pass], axis=0)

        self.lp = torch.nn.Conv2d(
            in_channels=3, out_channels=3, stride=1, kernel_size=5, padding=2, groups=3, bias=False)

        self.lp.weight.data = torch.Tensor(lp_filters).unsqueeze(1)

        # Freezing weights
        self.lp.weight.requires_grad = not freeze_weights

    def forward(self, x):

        o = self.lp(x)
        # o = o + torch.rand_like(o, device=o.device) * 16./255 - 8./255
        o = TSQuantization(o, filters=self.lp.weight, epsilon=self.beta*8.0/255)
        # breakpoint()

        return o


class DoG_LP_Layer(nn.Module):

    def __init__(self, beta=1.0, freeze_weights=True):
        super().__init__()
        self.lp = LowpassLayer(beta=beta)
        self.dog = DoGLayer()

    def forward(self, x):

        o = self.lp(x)
        o = self.dog(o)
        # o = TSQuantization(o, filters=self.lp.weight, epsilon=self.beta*8.0/255)
        # breakpoint()

        return o
