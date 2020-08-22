"""

Example Run

python -m pytorch-tutorials.cifar.main_combined --model ResNetEmbedding --epochs 100 -tr -sm

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
from ..models import ResNet, AttentionResNet, ConvolutionalAttentionResNet, SpatialAttentionResNet, MultiModeEmbeddingMaxpooling, MultiModeEmbeddingClassification, ConvolutionalSpatialAttentionResNet, VGG, MobileNet, MobileNetV2, PreActResNet, ResNetEmbedding
from ..train_test import adversarial_epoch, adversarial_test
from ..read_datasets import cifar10, cifar10_hard
from .parameters import get_arguments


logger = logging.getLogger(__name__)


def main():
    """ main function to run the experiments """

    args = get_arguments()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.directory + 'logs'):
        os.mkdir(args.directory + 'logs')

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(args.directory + 'logs/' + args.model +
                                '_' + args.tr_attack + '.log'),
            logging.StreamHandler()
            ])
    logger.info(pformat(vars(args)))
    logger.info("\n")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, test_loader = cifar10(args)
    hard_loader = cifar10_hard(args)
    x_min = 0.0
    x_max = 1.0

    # Decide on which model to use
    model = globals()[args.model]()
    model2 = globals()[args.model]()
    new_model = MultiModeEmbeddingMaxpooling(num_modes=2, input_size=64)

    # Checkpoint Namer
    checkpoint_name = args.model + ".pt"

    model.load_state_dict(torch.load(args.directory + "checkpoints/" + checkpoint_name))
    model2.load_state_dict(torch.load(args.directory + "checkpoints/" +
                                      "secondary_" + checkpoint_name))

    for i in range(10):
        new_model.parallel_layers[i].weight[0] = model.linear.weight[i]
        new_model.parallel_layers[i].weight[1] = model2.linear.weight[i]

    test_args = dict(model=model,
                     test_loader=test_loader)
    test_loss, test_acc = adversarial_test(**test_args)
    logger.info(f'Before: Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    model.linear = new_model
    model.to(device)

    test_args = dict(model=model,
                     test_loader=test_loader)
    test_loss, test_acc = adversarial_test(**test_args)
    logger.info(f'After: Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')


if __name__ == "__main__":
    main()
