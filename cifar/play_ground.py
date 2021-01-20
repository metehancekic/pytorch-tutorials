"""

Example Run

python -m pytorch-tutorials.cifar.main -at -bb -b 16 --epochs 100 --attack PGD --frontend LP_Gabor_Layer_v3 --model ResNet -tr -sm
"""
import time
import os
from pprint import pformat
from os import path
from tqdm import tqdm
import numpy as np
import logging

# Torch
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT
from deepillusion.torchattacks.analysis import whitebox_test
from deepillusion.torchattacks.analysis.plot import loss_landscape
# from deepillusion.torchdefenses import adversarial_epoch

# CIFAR10 TRAIN TEST CODES
from ..nn_tools import NeuralNetwork
from ..read_datasets import cifar10_black_box
from .initializer import initialize_everything
# from .gabor_trial import plot_image


def main():

    logger, args, device, train_loader, test_loader, data_params = initialize_everything()

    #--------------------------------------------------#
    #---------- Set up Model and checkpoint name ------#
    #--------------------------------------------------#
    from ..models import ResNet, VGG, VGG_modified, VGG_modified2

    checkpoint_name = args.model + ".pt"
    if args.frontend == "Identity":
        model = locals()[args.model]().to(device)
    else:
        from ..models.custom_layers import CenterSurroundModule, AutoEncoder, Decoder, CenterSurroundConv, DoGLayer, DoGLowpassLayer, LowpassLayer, DoG_LP_Layer, LP_Gabor_Layer, LP_Gabor_Layer_v2, LP_Gabor_Layer_v3, LP_Gabor_Layer_v4,  LP_Gabor_Layer_v5,  LP_Gabor_Layer_v6, LP_Layer, Identity
        frontend = locals()[args.frontend](beta=args.beta, BPDA_type=args.bpda_type).to(device)
        CNN = locals()[args.model]().to(device)
        model = AutoEncoder(frontend, CNN).to(device)
        checkpoint_name = args.frontend + "_beta_" + str(int(args.beta)) + checkpoint_name

    if args.tr_epoch_type == "Trades":
        checkpoint_name = "trades" + "_" + checkpoint_name
    elif args.tr_attack != "Standard":
        checkpoint_name = args.tr_attack + "_" + checkpoint_name

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    breakpoint()


if __name__ == "__main__":
    main()
