import torch
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np
from os import path


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
