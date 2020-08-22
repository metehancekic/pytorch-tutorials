
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader

from ..models import ResNet, AttentionResNet, ConvolutionalAttentionResNet, SpatialAttentionResNet, MultiModeEmbeddingClassification, ConvolutionalSpatialAttentionResNet, VGG, MobileNet, MobileNetV2, PreActResNet, ResNetEmbedding
from ..train_test import adversarial_epoch, adversarial_test
from ..read_datasets import cifar10_hard
from .parameters import get_arguments


def cifar10(args):
    """Returns the loader for train and test with given arguments."""

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])

    trainset = datasets.CIFAR10(
        root=args.directory + 'data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=1000, shuffle=True, **kwargs)

    testset = datasets.CIFAR10(root=args.directory + 'data',
                               train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def generate_hard_dataset():

    args = get_arguments()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = cifar10(args)
    x_min = 0.0
    x_max = 1.0

    # Decide on which model to use
    model = globals()[args.model]().to(device)

    if device == "cuda":
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    checkpoint_name = args.model + ".pt"

    model.load_state_dict(torch.load(args.directory + "checkpoints/" + checkpoint_name))

    print("Clean test accuracy")
    test_args = dict(model=model,
                     test_loader=test_loader)
    test_loss, test_acc = adversarial_test(**test_args)
    print(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    model.eval()

    device = model.parameters().__next__().device

    hard_dataset = []
    hard_labels = []

    for data, target in train_loader:

        data, target = data.to(device), target.to(device)

        output, embedding = model(data)

        pred_adv = output.argmax(dim=1, keepdim=True).squeeze()

        hard_dataset.append(data[pred_adv != target].detach().cpu().numpy())
        hard_labels.append(target[pred_adv != target].detach().cpu().numpy())

    hard_dataset = np.concatenate(tuple(hard_dataset), axis=0)
    hard_labels = np.concatenate(tuple(hard_labels), axis=0)

    np.savez(args.directory + "data/hard_examples", hard_dataset, hard_labels)


def main():

    hard_loader = cifar10_hard(args)
    data, target = iter(hard_loader).__next__()
    for i in range(10):
        print(target[i])
        plt.figure()
        plt.imshow(data.permute(0, 2, 3, 1).detach().cpu().numpy()[i])
    plt.show()


if __name__ == '__main__':

    main()
