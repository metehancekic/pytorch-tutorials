"""
Neural Network models for training and testing implemented in PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=3, out_channels=16, stride=1, kernel_size=3, padding=1, bias=True)
        self.conv2 = torch.nn.ConvTranspose2d(
            in_channels=16, out_channels=64, stride=1, kernel_size=3, padding=1, bias=True)
        self.conv3 = torch.nn.ConvTranspose2d(
            in_channels=64, out_channels=16, stride=1, kernel_size=3, padding=1, bias=True)
        self.conv4 = torch.nn.ConvTranspose2d(
            in_channels=16, out_channels=3, stride=1, kernel_size=3, padding=1, bias=True)

    def forward(self, x):

        o = F.relu(self.conv1(x))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        o = self.conv4(o)

        return o  # .clamp(0.0, 1.0)
