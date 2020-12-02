
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...utils import l1_normalizer, DifferenceOfGaussian2d, rgb2yrgb, rgb2y
from . import DReLU, DTReLU, TQuantization, TSQuantization, take_top_coeff, GaborConv2d, take_top_coeff_BPDA


class LP_Gabor_Layer(nn.Module):

    def __init__(self, beta=1.0, BPDA_type="maxpool_like", freeze_weights=True):
        super().__init__()

        self.beta = beta

        self.low_pass = l1_normalizer(np.ones((5, 5)))
        lp_filters = np.stack([self.low_pass, self.low_pass, self.low_pass], axis=0)
        self.lp = torch.nn.Conv2d(
            in_channels=3, out_channels=3, stride=1, kernel_size=5, padding=2, groups=3, bias=False)
        self.lp.weight.data = torch.Tensor(lp_filters).unsqueeze(1)
        # Freezing weights
        self.lp.weight.requires_grad = not freeze_weights

        self.gabor_layer = GaborConv2d(in_channels=3,
                                       out_channels=128, padding=5, kernel_size=(11, 11))
        self.gabor_layer.calculate_weights()

        self.to_img = torch.nn.Conv2d(
            in_channels=128, out_channels=3, stride=1, kernel_size=5, padding=2, bias=False)
        self.set_BPDA_type(BPDA_type)

    def set_BPDA_type(self, BPDA_type="maxpool_like"):
        self.BPDA_type = BPDA_type
        if self.BPDA_type == "maxpool_like":
            self.take_top = take_top_coeff
        elif self.BPDA_type == "identity":
            self.take_top = take_top_coeff_BPDA().apply

        # self.gabor_layer.weight.data =

    def forward(self, x):

        o = self.lp(x)
        if self.training:
            o = o + torch.rand_like(o, device=o.device) * 16./255 - 8./255
        o = TSQuantization(o, filters=self.lp.weight, epsilon=self.beta*8.0/255)
        o = self.gabor_layer(o)
        o = self.take_top(o)
        o = TSQuantization(o, filters=self.gabor_layer.weight, epsilon=8.0/255)
        o = self.to_img(o)

        return o


class LP_Gabor_Layer_v2(nn.Module):
    # Best
    def __init__(self, beta=1.0, BPDA_type="maxpool_like", freeze_weights=True):
        super().__init__()

        self.beta = beta

        self.low_pass = l1_normalizer(np.ones((5, 5)))
        lp_filters = np.stack([self.low_pass, self.low_pass, self.low_pass], axis=0)
        self.lp = torch.nn.Conv2d(
            in_channels=3, out_channels=3, stride=1, kernel_size=5, padding=2, groups=3, bias=False)
        self.lp.weight.data = torch.Tensor(lp_filters).unsqueeze(1)
        # Freezing weights
        self.lp.weight.requires_grad = not freeze_weights

        self.gabor_layer = GaborConv2d(in_channels=3,
                                       out_channels=128, padding=5, kernel_size=(11, 11))
        self.gabor_layer.calculate_weights()

        self.to_img = torch.nn.Conv2d(
            in_channels=128, out_channels=3, stride=1, kernel_size=5, padding=2, bias=False)
        self.set_BPDA_type(BPDA_type)

    def set_BPDA_type(self, BPDA_type="maxpool_like"):
        self.BPDA_type = BPDA_type
        if self.BPDA_type == "maxpool_like":
            self.take_top = take_top_coeff
        elif self.BPDA_type == "identity":
            self.take_top = take_top_coeff_BPDA().apply

    def forward(self, x):

        o = self.lp(x)
        if self.training:
            o = o + torch.rand_like(o, device=o.device) * 16./255 - 8./255
        o = DTReLU(o, filters=self.lp.weight, epsilon=self.beta*8.0/255)
        o = self.gabor_layer(o)
        o = self.take_top(o)
        # if self.training:
        #     o = o + torch.rand_like(o, device=o.device) * 16./255 - 8./255
        o = TSQuantization(o, filters=self.gabor_layer.weight, epsilon=8.0/255)
        o = self.to_img(o)

        return o
