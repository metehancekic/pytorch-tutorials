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


from ..models import LeNet, LeNet2d
from ..train_test import adversarial_epoch, adversarial_test
from ..read_datasets import mnist
from .parameters import get_arguments

# from models import ConvolutionalNeuralNet as CNN
# from arguments import get_arguments_mnist

logger = logging.getLogger(__name__)


def main():

    # Get arguments for current run
    args = get_arguments()

    # Set seeds to get same results every run
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Create a log fiolder to save logs
    if not os.path.exists(args.directory + 'logs'):
        os.mkdir(args.directory + 'logs')

    # Logging initiator
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

    # Check if GPU exists and if you want to use it
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Get train and test loader
    train_loader, test_loader = mnist(args)
    x_min = 0.0  # min pixel value
    x_max = 1.0  # max pixel value

    # Define and move the model to device
    model = globals()[args.model]().to(device)
    breakpoint()

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

    # Parameters of the data for adversarial attack function
    data_params = {"x_min": x_min, "x_max": x_max}

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

    # Train network if args.train is set to True (You can set that true by calling '-tr' flag, default is False)
    if args.train:
        logger.info(args.tr_attack + " training")
        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            # Training (Adversarial or Standard)
            train_args = dict(model=model,
                              train_loader=train_loader,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              adversarial_args=adversarial_args)
            train_loss, train_acc = adversarial_epoch(**train_args)

            # Testing (Standard)
            test_args = dict(model=model,
                             test_loader=test_loader)
            test_loss, test_acc = adversarial_test(**test_args)

            end_time = time.time()
            lr = scheduler.get_lr()[0]
            logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}')
            logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

        # Save model parameters
        if args.save_model:
            if not os.path.exists(args.directory + "checkpoints/"):
                os.makedirs(args.directory + "checkpoints/")
            torch.save(model.state_dict(), args.directory + "checkpoints/" + checkpoint_name)

    else:
        model.load_state_dict(torch.load(args.directory + "checkpoints/" + checkpoint_name))

        logger.info("Clean test accuracy")
        test_args = dict(model=model,
                         test_loader=test_loader)
        test_loss, test_acc = adversarial_test(**test_args)
        logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    if args.attack_network:
        attack_params = {
            "norm": args.norm,
            "eps": args.epsilon,
            "alpha": args.alpha,
            "step_size": args.step_size,
            "num_steps": args.num_iterations,
            "random_start": args.rand,
            "num_restarts": args.num_restarts,
            "ensemble_size": 10
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

        test_args = dict(model=model,
                         test_loader=test_loader,
                         adversarial_args=adversarial_args,
                         verbose=True,
                         progress_bar=True)
        test_loss, test_acc = adversarial_test(**test_args)
        logger.info(f'{args.attack} test \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}\n')


if __name__ == '__main__':
    main()
