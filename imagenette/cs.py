"""

Example Run

python -m deep_noise_rejection.CIFAR10.main --model ResNetMadry -tra RFGSM -at -Ni 7 -tr -sm

"""
import time
import os
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
# from deepillusion.torchdefenses import adversarial_epoch

# CIFAR10 TRAIN TEST CODES
from ..models import ResNet, VGG, MobileNet, MobileNetV2, PreActResNet, EfficientNet
from ..models.custom_layers import CenterSurroundModule, AutoEncoder, Decoder, CenterSurroundConv, DoGLayer, DoGLowpassLayer, LowpassLayer, DoG_LP_Layer, LP_Gabor_Layer, LP_Gabor_Layer_v2
from ..train_test import adversarial_epoch, adversarial_test, reconstruction_epoch, reconstruction_test, frontend_outputs
from ..read_datasets import imagenette, imagenette_black_box
from .parameters import get_arguments
# from .gabor_trial import plot_image


logger = logging.getLogger(__name__)


def main():
    """ main function to run the experiments """

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
            logging.FileHandler(args.directory + 'logs/' + "LowPassFrontEnd" + '.log'),
            logging.StreamHandler()
            ])
    logger.info(args)
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
    #------------ Read Data and Set Model -------------#
    #--------------------------------------------------#
    train_loader, test_loader = imagenette(args)
    x_min = 0.0
    x_max = 1.0
    # frontend = CenterSurroundConv(beta=args.beta).to(device)
    # frontend = DoGLayer(beta=args.beta).to(device)
    frontend = globals()[args.frontend](beta=args.beta, BPDA_type=args.bpda_type).to(device)
    # img, lbl = test_loader.__iter__().__next__()
    # img = img.to(device)
    # frontend(img)
    # breakpoint()
    CNN = globals()[args.model]().to(device)
    model = AutoEncoder(frontend, CNN).to(device)
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
                   RFGSM=RFGSM,
                   PGD_EOT=PGD_EOT)

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

    # scheduler = None
    # Checkpoint Namer
    checkpoint_name = "LowPass_Gabor_CNN_bpda_sternary_b_" + str(int(args.beta))

    if args.train:
        logger.info("Standard training")
        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            train_args = dict(model=model,
                              train_loader=train_loader,
                              optimizer=optimizer,
                              scheduler=scheduler)
            train_loss, train_acc = adversarial_epoch(**train_args)

            test_args = dict(model=model,
                             test_loader=test_loader)
            test_loss, test_acc = adversarial_test(**test_args)

            end_time = time.time()
            lr = scheduler.get_lr()[0]
            logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}')
            logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

        # Save model parameters
        if not os.path.exists(args.directory + "checkpoints/frontends/"):
            os.makedirs(args.directory + "checkpoints/frontends/")
        torch.save(model.state_dict(), args.directory +
                   "checkpoints/frontends/" + checkpoint_name)

    else:
        model.load_state_dict(torch.load(
            args.directory + "checkpoints/frontends/" + checkpoint_name))

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

        test_args = dict(model=model,
                         test_loader=test_loader,
                         adversarial_args=adversarial_args,
                         verbose=True,
                         progress_bar=True)
        test_loss, test_acc = adversarial_test(**test_args)
        logger.info(f'{args.attack} test \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}\n')

    if args.black_box:
        attack_loader = imagenette_black_box(args)

        test_args = dict(model=model,
                         test_loader=attack_loader)
        test_loss, test_acc = adversarial_test(**test_args)
        logger.info("Black Box test accuracy")
        logger.info(f'Blackbox Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')


if __name__ == "__main__":
    main()
