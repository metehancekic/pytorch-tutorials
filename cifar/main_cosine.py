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
from ..models import *
from ..train_test import adversarial_epoch, adversarial_test, cosine_epoch  # , embedding_analysis
from ..read_datasets import cifar10
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
    x_min = 0.0
    x_max = 1.0

    # Decide on which model to use
    model = globals()[args.model]().to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f" Number of total trainable parameters: {params}")
    # breakpoint()

    if device == "cuda":
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    logger.info(model)
    logger.info("\n")

    # Which optimizer to be used for training
    optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min,
                                                  max_lr=args.lr_max, step_size_up=lr_steps/2,
                                                  step_size_down=lr_steps/2)
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
    checkpoint_name += "cos1000.pt"

    # Train network if args.train is set to True (You can set that true by calling '-tr' flag, default is False)
    if args.train:
        logger.info(args.tr_attack + " training")
        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()

            train_args = dict(model=model,
                              train_loader=train_loader,
                              optimizer=optimizer,
                              scheduler=scheduler,
                              adversarial_args=adversarial_args)
            train_loss, train_acc = cosine_epoch(**train_args)

            test_args = dict(model=model,
                             test_loader=test_loader)
            test_loss, test_acc = adversarial_test(**test_args)

            end_time = time.time()
            lr = scheduler.get_lr()[0]
            logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}')
            logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

            if epoch % args.tr_attack_logging == 0:

                adversarial_args_test = dict(attack=attacks[args.attack],
                                             attack_args=dict(net=model,
                                                              data_params=data_params,
                                                              attack_params=attack_params,
                                                              loss_function="cross_entropy",
                                                              verbose=False,
                                                              progress_bar=True))
                test_args = dict(model=model,
                                 test_loader=test_loader,
                                 adversarial_args=adversarial_args_test,
                                 verbose=True,
                                 progress_bar=True)
                test_loss, test_acc = adversarial_test(**test_args)
                logger.info(f'Adversarial Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

        # Save model parameters
        if args.save_model:
            if not os.path.exists(args.directory + "checkpoints/"):
                os.makedirs(args.directory + "checkpoints/")
            torch.save(model.state_dict(), args.directory + "checkpoints/" + checkpoint_name)

    else:
        model.load_state_dict(torch.load(args.directory + "checkpoints/" + checkpoint_name))
        # model.load_state_dict(torch.load(args.directory + "checkpoints/" + "model-res-epoch76.pt"))

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

    # if args.black_box:
    #     attack_loader = cifar10_black_box(args)

    #     test(model, attack_loader)


if __name__ == "__main__":
    main()
