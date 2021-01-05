"""
Neural Network models for training and testing implemented in PyTorch
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from ..tools import Normalize


class L1LeNet(nn.Module):

    # 2 Conv layers, 2 Fc layers
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()

        # self.norm = Normalize(mean=[0.1307], std=[0.3081])
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5,
                               stride=1, padding=2, bias=True)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024, bias=True)
        self.fc2 = nn.Linear(1024, num_classes, bias=True)

    def forward(self, x):

        # out = self.norm(x)
        self.l1_normalize_weights()
        out = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        out = F.max_pool2d(F.relu(self.conv2(out)), (2, 2))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out

    def l1_normalize_weights(self):

        self.conv1.weight = Parameter(self.conv1.weight / torch.norm(self.conv1.weight.view(
            self.conv1.weight.size(0), -1), p=1, dim=1, keepdim=True).unsqueeze(1).unsqueeze(1))

        self.conv2.weight = Parameter(self.conv2.weight / torch.norm(self.conv2.weight.view(
            self.conv2.weight.size(0), -1), p=1, dim=1, keepdim=True).unsqueeze(1).unsqueeze(1))

        self.fc1.weight = Parameter(
            self.fc1.weight / torch.norm(self.fc1.weight, p=1, dim=1, keepdim=True))
        self.fc2.weight = Parameter(
            self.fc2.weight / torch.norm(self.fc2.weight, p=1, dim=1, keepdim=True))
