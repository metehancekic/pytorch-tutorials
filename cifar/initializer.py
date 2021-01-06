"""

Example Run

python -m pytorch-tutorials.cifar.main -at -bb -b 16 --epochs 100 --attack PGD --frontend LP_Gabor_Layer_v3 --model ResNet -tr -sm
"""

import os
from pprint import pformat
import numpy as np
import logging

# Torch
import torch
from torch.optim.lr_scheduler import MultiStepLR


# CIFAR10 TRAIN TEST CODES
from ..nn_tools import NeuralNetwork
from ..read_datasets import cifar10, cifar10_black_box
from .parameters import get_arguments
# from .gabor_trial import plot_image


def initialize_everything():
    """ main function to run the experiments """
    logger = logging.getLogger(__name__)
    args = get_arguments()

    #--------------------------------------------------#
    #-------------- LOGGER INITIALIZATION -------------#
    #--------------------------------------------------#
    if not os.path.exists(args.directory + 'logs'):
        os.mkdir(args.directory + 'logs')

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.directory + 'logs/' +
                                args.frontend + "_" + args.model + '.log'),
            logging.StreamHandler()
            ])
    logger.info(pformat(vars(args)))
    logger.info("\n")

    #--------------------------------------------------#
    #--------------- Seeds and Device -----------------#
    #--------------------------------------------------#
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #--------------------------------------------------#
    #------------------ Read Data ---------------------#
    #--------------------------------------------------#
    train_loader, test_loader = cifar10(args)
    x_min = 0.0
    x_max = 1.0
    data_params = {"x_min": x_min, "x_max": x_max}

    return logger, args, device, train_loader, test_loader, data_params
