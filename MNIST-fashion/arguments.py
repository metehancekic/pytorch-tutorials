'''
Hyper-parameters.
'''

import argparse

def get_arguments_mnist():

    parser = argparse.ArgumentParser(description='PyTorch MNIST ')
    
    parser.add_argument('--batch-size', type=int, default=100, metavar='N', help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N', help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='mnist', metavar='mnist/fashion', help='Which dataset to use (default: mnist)')
    parser.add_argument('-sm', '--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('-im', '--initialize-model', action='store_true', default=True, help='Initialize the Model from checkpoint with standard parameters')
    
    # Use this to train or attack neural network
    parser.add_argument('-tr', '--train_network', action='store_true', help='Train network, default = False')

    args = parser.parse_args()

    return args