"""

Example Run

python -m pytorch-tutorials.MNIST-fashion.main --model ResNetMadry -tra RFGSM -at -Ni 7 -tr -sm

"""

import numpy as np
import logging
from pprint import pformat
import time
import os
from os import path
from tqdm import tqdm


# Torch
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD
from deepillusion.torchattacks.analysis import whitebox_test
# from deepillusion.torchdefenses import adversarial_epoch

# CIFAR10 TRAIN TEST CODES
from ..models import ResNet, AttentionResNet, ConvolutionalAttentionResNet, SpatialAttentionResNet, MultiModeEmbeddingClassification, ConvolutionalSpatialAttentionResNet, VGG, MobileNet, MobileNetV2, PreActResNet, ResNetEmbedding
from ..train_test import adversarial_epoch, adversarial_test
from ..read_datasets import cifar10
from .parameters import get_arguments


def embedding_analysis(model, test_loader, adversarial_args=None, verbose=False, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        adversarial_args :                   (dict)
            attack:                          (deepillusion.torchattacks)
            attack_args:                     (dict)
                attack arguments for given attack except "x" and "y_true"
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device

    model.eval()

    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    all_embeddings = []
    all_pred = []
    all_labels = []
    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        output, embedding = model(data)

        pred = output.argmax(dim=1, keepdim=True)

        all_embeddings.append(embedding.detach().cpu().numpy())
        all_pred.append(pred.squeeze().detach().cpu().numpy())
        all_labels.append(target.detach().cpu().numpy())

        # from matplotlib import pyplot as plt
        # cm = plt.get_cmap('gist_rainbow')
        # plt.figure()
        # for i in range(10):
        #     plt.scatter(embedding.detach().cpu().numpy()[(target.detach().cpu().numpy() == i)[:, 0]][:, 0],
        #                 embedding.detach().cpu().numpy()[(target.detach().cpu().numpy() == i)[:, 0]][:, 1], s=1, c=cm(1.*i/10))
        # plt.figure()
        # for i in range(10):
        #     plt.scatter(embedding.detach().cpu().numpy()[(target.detach().cpu().numpy() == i)][:, 0],
        #                 embedding.detach().cpu().numpy()[(target.detach().cpu().numpy() == i)][:, 1], s=1, c=cm(1.*i/10))
        # plt.show()
        # breakpoint()

    all_embeddings = np.concatenate(tuple(all_embeddings), axis=0)
    all_pred = np.concatenate(tuple(all_pred), axis=0)
    all_labels = np.concatenate(tuple(all_labels), axis=0)

    # breakpoint()

    return all_embeddings, all_pred, all_labels


def main():
    """ main function to run the experiments """

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
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    attacks = dict(Standard=None,
                   PGD=PGD,
                   FGSM=FGSM,
                   RFGSM=RFGSM)

    attack_params = {
        "norm": args.tr_norm,
        "eps": args.tr_epsilon,
        "alpha": args.tr_alpha,
        "step_size": args.tr_step_size,
        "num_steps": args.tr_num_iterations,
        "random_start": args.tr_rand,
        "num_restarts": args.tr_num_restarts,
        }

    data_params = {"x_min": x_min, "x_max": x_max}

    adversarial_args = dict(attack=attacks[args.tr_attack],
                            attack_args=dict(net=model,
                                             data_params=data_params,
                                             attack_params=attack_params,
                                             verbose=False))

    # Checkpoint Namer
    checkpoint_name = args.model
    if adversarial_args["attack"]:
        checkpoint_name += "_adv_" + args.tr_attack
    checkpoint_name += ".pt"

    model.load_state_dict(torch.load(args.directory + "checkpoints/" + checkpoint_name))

    test_args = dict(model=model,
                     test_loader=test_loader)
    all_embeddings, all_pred, all_labels = embedding_analysis(**test_args)
    breakpoint()


if __name__ == "__main__":
    main()
