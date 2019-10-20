'''
Neural Network models for training and testing implemented in PyTorch
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib as mpl



# Standard 4 layer CNN implementation
class ConvolutionalNeuralNet(nn.Module):

    # 2 Conv layers, 2 Fc layers, 1 output layer

    def __init__(self, args):
        super(ConvolutionalNeuralNet, self).__init__()

        # All hyperparameters in args
        self.args = args

        # 1 input image channel, 20 output channels, 5x5 square convolution
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv1 = nn.Conv2d(1, 20, 5, 1, bias = True)
        self.conv2 = nn.Conv2d(20, 50, 5, 1, bias = True)
        self.fc1 = nn.Linear(4 * 4 * 50, 1000, bias = True)
        # torch.nn.Linear(in_features, out_features, bias=True)
        self.fc2 = nn.Linear(1000, 1000, bias = True)
        self.fc3 = nn.Linear(1000, 10)


    # If you want to see sparsity of neuron outputs for each layer, set show_sparsity = True
    def forward(self, x, show_sparsity = False):


        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        if show_sparsity:
            # To be able to compute sparsity (or any other metric using the node of graph)
            # .data : gets data (Optional)
            # .detach : Detaches from the graph
            # .cpu : moves it to CPU
            # .clone : similar .copy in np
            # .numpy : converts it to numpy array
            print('\nSparsity of first layer : {:.3f}'.format(np.count_nonzero(x.data.detach().cpu().clone().numpy())/x.data.numel()))

        # Max Pooling with [2,2] strides
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        if show_sparsity:
            print('Sparsity of second layer : {:.3f}'.format(np.count_nonzero(x.data.detach().cpu().clone().numpy())/x.data.numel()))

        # Reshapes the tensor to feed in fully connected linear layer
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        
        if show_sparsity:
            print('Sparsity of third layer : {:.3f}'.format(np.count_nonzero(x.data.detach().cpu().clone().numpy())/x.data.numel()))
            
        x = F.relu(self.fc2(x))
        
        if show_sparsity:
            print('Sparsity of fourth layer : {:.3f}'.format(np.count_nonzero(x.data.detach().cpu().clone().numpy())/x.data.numel()))
        
        x = self.fc3(x)
        
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    