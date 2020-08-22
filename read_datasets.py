""" Loader functions for CIFAR10 dataset including train, test, blackbox test set """

import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader


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
        root=args.directory + 'data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, **kwargs)

    testset = datasets.CIFAR10(root=args.directory + 'data',
                               train=False, download=True, transform=transform_test)
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


def cifar10_hard(args):

    hard_dataset = np.load(args.directory + "data/hard_examples.npz")['arr_0']
    hard_labels = np.load(args.directory + "data/hard_examples.npz")['arr_1']

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    tensor_x = torch.Tensor(hard_dataset)  # transform to torch tensor
    tensor_y = torch.Tensor(hard_labels).type(torch.long)

    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    # create your dataloader
    hard_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, **kwargs)

    return hard_loader


def mnist(args):

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    if args.dataset == "mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.directory + "data",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=args.train_batch_size,
            shuffle=True,
            **kwargs
            )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                args.directory + "data",
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs
            )

    elif args.dataset == "fashion":
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                args.directory + "data",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
                ),
            batch_size=args.train_batch_size,
            shuffle=True,
            **kwargs
            )

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(
                args.directory + "data",
                train=False,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(), ]),
                ),
            batch_size=args.test_batch_size,
            shuffle=True,
            **kwargs
            )

    return train_loader, test_loader


def imagenette(args):

    data_dir = args.directory + "data/"
    train_dir = path.join(data_dir, "train")
    test_dir = path.join(data_dir, "val")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 2, "pin_memory": True} if use_cuda else {}

    transform_train = transforms.Compose([
        transforms.RandomCrop((160), padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(160),
        transforms.ToTensor(),
        ])

    trainset = datasets.ImageFolder(train_dir, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
        )

    return train_loader, test_loader


def imagenette_blackbox(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # Read
    if args.attack_otherbox_type == "B-T":
        filename = "B-T.npy"
    elif args.attack_otherbox_type == "PW-T":
        filename = "PW-T.npy"

    test_blackbox = np.load(args.directory + "data/attacked_dataset/" + args.dataset + filename)

    data_dir = args.directory + "data/"
    test_dir = path.join(data_dir, "original_dataset", "imagenette2-160", "val")
    transform_test = transforms.Compose([transforms.ToTensor(), ])
    testset = datasets.ImageFolder(test_dir, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2
        )

    tensor_x = torch.Tensor(test_blackbox / np.max(test_blackbox))
    tensor_y = torch.Tensor(test_loader.dataset.targets).long()

    tensor_data = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    attack_loader = torch.utils.data.DataLoader(
        tensor_data, batch_size=args.test_batch_size, shuffle=False, **kwargs
        )

    return attack_loader
