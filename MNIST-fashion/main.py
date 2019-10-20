'''
Author: Metehan Cekic
Main running script for testing and training of mnist classification CNN model implemented with PyTorch

Example Run

python main.py -tr -sm --dataset fashion 


'''


from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from matplotlib import pyplot as plt
import os

from models import ConvolutionalNeuralNet as CNN
from arguments import get_arguments_mnist




# Train network
def train(args, model, device, train_loader, optimizer, epoch):
    
    # Set phase to training
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        
        # Feed data to device ( e.g, GPU )
        data, target = data.to(device), target.to(device)
        
        # Each iteration clean gradient values
        optimizer.zero_grad()

        # Forward graph
        output = model(data)

        # Call Loss Object 
        cross_ent = torch.nn.CrossEntropyLoss()

        # Create loss with given loss object
        loss = cross_ent(output, target)

        # Compute loss
        loss.backward()

        # Optimize according to
        optimizer.step()


        # Print out loss and accuracy for training at each log_interval times
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Test Network
def test(args, model, device, test_loader):
    
    # Set phase to testing
    model.eval()

    # Initialize test_loss and correct
    test_loss = 0
    correct = 0
    
    # We don't have to compute gradients
    # Disables gradient tracking
    with torch.no_grad():
        for data, target in test_loader:

            # Feed data to device ( e.g, GPU )
            data, target = data.to(device), target.to(device)

            # Forward graph
            output = model(data)

            # Sum up batch loss in test set
            test_loss += F.cross_entropy(output, target, reduction='sum').item() 

            # Get the index of predictions
            pred = output.argmax(dim=1, keepdim=True) 

            # Count the number of the predictions which are correct
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Divide summed-up loss by the number of datapoints in dataset
    test_loss /= len(test_loader.dataset)

    # Print out Loss and Accuracy for test set
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    
    # Get arguments for current run
    args = get_arguments_mnist()

    # Check if GPU exists and if you want to use it
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Get same results for each training with same parameters !! 
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    

    # Decide on which dataset to use
    # You can normalize, transform dataset before using it over here
    if args.dataset == 'mnist':
        
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor()])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)

    elif args.dataset == 'fashion':
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor()
                                  ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)


    # Define and move the model to device
    model = CNN(args).to(device)

    # Which optimizer to be used for training
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    

    # Initialize model with standard model with layers without biases
    if args.initialize_model:
        model.load_state_dict(torch.load("checkpoints/cnn_" + args.dataset + ".pt"))
        test(args, model, device, test_loader)

    # Train network if args.train_network is set to True ( You can set that true by calling '-tr' flag, default is False )
    if args.train_network:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch-1)
            test(args, model, device, test_loader)

        # Save model parameters
        if not os.path.exists("checkpoints/"):
            os.makedirs("checkpoints/")

        if args.save_model:
            torch.save(model.state_dict(),"checkpoints/cnn_" + args.dataset + ".pt")

    
        
if __name__ == '__main__':
    main()


