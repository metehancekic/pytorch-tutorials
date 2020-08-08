""" Loader functions for CIFAR10 dataset including train, test, blackbox test set """

import numpy as np

import torch
from torchvision import datasets, transforms


def cifar10(args):
    """Returns the loader for train and test with given arguments."""

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

    trainset = datasets.CIFAR10(
        root=args.directory + 'data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, **kwargs)

    testset = datasets.CIFAR10(root=args.directory + 'data',
                               train=False, download=False, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def cifar10_black_box(args):
    """Returns the loader for blackbox attacked test loader with given arguments."""

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Read
    test_blackbox = np.load(
        args.directory + 'data/attacked_dataset/madry_AT_attack.npy')

    testset = datasets.CIFAR10(root=args.directory + 'data', train=False,
                               transform=None, target_transform=None, download=False, )

    y_test = np.array(testset.targets)

    tensor_x = torch.Tensor(
        test_blackbox/np.max(test_blackbox)).permute(0, 3, 1, 2)
    tensor_y = torch.Tensor(y_test).long()

    tensor_data = torch.utils.data.TensorDataset(
        tensor_x, tensor_y)  # create your datset
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    return attack_loader
