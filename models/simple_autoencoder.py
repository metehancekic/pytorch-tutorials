import torch
from torch import nn
import torch.nn.functional as F


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3, stride=1, padding=0, bias=True
            )
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=1, padding=0, bias=True
            )
        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=0, bias=True
            )

        self.deconv1 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=1, padding=0, bias=True
            )
        self.deconv2 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=1, padding=0, bias=True
            )
        self.deconv3 = nn.ConvTranspose2d(
            32, 3, kernel_size=3, stride=1, padding=0, bias=True
            )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.deconv1(out))
        out = F.relu(self.deconv2(out))
        out = F.relu(self.deconv3(out))
        return out.clamp(0.0, 1.0)
