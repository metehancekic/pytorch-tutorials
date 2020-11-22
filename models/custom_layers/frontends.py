
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...utils import l1_normalizer, DifferenceOfGaussian2d, rgb2yrgb, rgb2y
from . import DReLU, DTReLU, TQuantization, TSQuantization, take_top_coeff, GaborConv2d


class LP_Gabor_Layer(nn.Module):

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

        self.gabor_layer = GaborConv2d(in_channels=3,
                                       out_channels=128, kernel_size=(11, 11))
        self.gabor_layer.calculate_weights()

        # self.gabor_layer.weight.data =

    def forward(self, x):

        o = self.lp(x)
        o = o + torch.rand_like(o, device=o.device) * 16./255 - 8./255
        o = TSQuantization(o, filters=self.lp.weight, epsilon=self.beta*8.0/255)

        o = self.gabor_layer(o)
        breakpoint()
        o = take_top_coeff(o)
        breakpoint()
        o = TSQuantization(o, filters=self.lp.weight, epsilon=self.beta*8.0/255)
        breakpoint()

        return o