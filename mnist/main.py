'''
Author: Metehan Cekic
Main running script for testing and training of mnist classification CNN model implemented with PyTorch

Example Run

python main.py -tr -sm --dataset fashion 


'''
import numpy as np
import logging
from pprint import pformat
import time
import os
from os import path
from tqdm import tqdm

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# ATTACK CODES
from deepillusion.torchattacks import FGSM, RFGSM, PGD
from deepillusion.torchattacks.analysis import whitebox_test

from ..nn_tools import NeuralNetwork
from ..train_test import adversarial_epoch, adversarial_test
from ..read_datasets import mnist
from .parameters import get_arguments

# from models import ConvolutionalNeuralNet as CNN
# from arguments import get_arguments_mnist


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
            logging.FileHandler(args.directory + 'logs/' + args.model + '.log'),
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
    train_loader, test_loader = mnist(args)
    x_min = 0.0
    x_max = 1.0
    data_params = {"x_min": x_min, "x_max": x_max}

    return logger, args, device, train_loader, test_loader, data_params


def main():

    logger, args, device, train_loader, test_loader, data_params = initialize_everything()

    from ..models import LeNet, LeNet2d
    from ..models.custom_layers import L1LeNet
    # Define and move the model to device
    model = locals()[args.model]().to(device)

    # Check the total number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info(f" Number of total trainable parameters: {params}")
    del model_parameters

    # Paralellize
    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    # Print Model summary
    logger.info(model)
    logger.info("\n")

    # Which optimizer to be used for training
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Cyclic learning Rate, first half linearly increases, then drops linearly
    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min,
                                                  max_lr=args.lr_max, step_size_up=lr_steps/2,
                                                  step_size_down=lr_steps/2)

    # Adversarial Attack dictionary for training, Standard means no adversarial training
    attacks = dict(Standard=None,
                   PGD=PGD,
                   FGSM=FGSM,
                   RFGSM=RFGSM)

    # Parameters for adversarial attack
    attack_params = {
        "norm": args.tr_norm,
        "eps": args.tr_epsilon,
        "alpha": args.tr_alpha,
        "step_size": args.tr_step_size,
        "num_steps": args.tr_num_iterations,
        "random_start": args.tr_rand,
        "num_restarts": args.tr_num_restarts,
        }

    # Adversarial arguments
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

    NN = NeuralNetwork(model, args.model, optimizer, scheduler)

    if args.train:
        NN.train_model(train_loader, test_loader, logger, epoch_type=args.tr_epoch_type, num_epochs=args.epochs,
                       log_interval=args.log_interval, adversarial_args=adversarial_args)

        if not os.path.exists(args.directory + "checkpoints/"):
            os.makedirs(args.directory + "checkpoints/")
        NN.save_model(checkpoint_dir=args.directory + "checkpoints/" + checkpoint_name)

    else:
        NN.load_model(checkpoint_dir=args.directory + "checkpoints/" + checkpoint_name)
        logger.info("Clean test accuracy")
        test_loss, test_acc = NN.eval_model(test_loader)
        logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    # if args.analyze_network:
    #     loss_landscape(model=model, data_loader=test_loader, img_index=0)
    #     outputs = frontend_analysis(model=frontend, test_loader=test_loader)
    #     breakpoint()

    # if args.black_box:
    #     attack_loader = cifar10_black_box(args)
    #     test_loss, test_acc = NN.eval_model(attack_loader)
    #     logger.info("Black Box test accuracy")
    #     logger.info(f'Blackbox Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    if args.attack_network:
        attack_params = {
            "norm": args.norm,
            "eps": args.epsilon,
            "alpha": args.alpha,
            "step_size": args.step_size,
            "num_steps": args.num_iterations,
            "random_start": args.rand,
            "num_restarts": args.num_restarts,
            "EOT_size": 10
            }

        adversarial_args = dict(attack=attacks[args.attack],
                                attack_args=dict(net=model,
                                                 data_params=data_params,
                                                 attack_params=attack_params,
                                                 loss_function="cross_entropy",
                                                 verbose=False,
                                                 progress_bar=True))

        for key in attack_params:
            logger.info(key + ': ' + str(attack_params[key]))

        test_loss, test_acc = NN.eval_model(
            test_loader, adversarial_args=adversarial_args, progress_bar=True, save_blackbox=False)
        logger.info(f'{args.attack} test \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}\n')


if __name__ == '__main__':
    main()
