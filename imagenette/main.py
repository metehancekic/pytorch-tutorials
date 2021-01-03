"""

Example Run

python -m pytorch-tutorials.imagenette.cs -at -bb -b 16 --epochs 100 --attack PGD --frontend LP_Gabor_Layer_v3 --model ResNet -tr -sm
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
    train_loader, test_loader = imagenette(args)
    x_min = 0.0
    x_max = 1.0
    data_params = {"x_min": x_min, "x_max": x_max}

    #--------------------------------------------------#
    #---------- Set up Model and checkpoint name ------#
    #--------------------------------------------------#
    from ..models import ResNet, VGG, MobileNet, MobileNetV2, PreActResNet, EfficientNet

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
    logger.info(model)
    logger.info("\n")

    #--------------------------------------------------#
    #------------ Optimizer and Scheduler -------------#
    #--------------------------------------------------#
    optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min,
                                                  max_lr=args.lr_max, step_size_up=lr_steps/2,
                                                  step_size_down=lr_steps/2)

    #--------------------------------------------------#
    #------------ Adversarial Argumenrs ---------------#
    #--------------------------------------------------#

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
        "beta": args.tr_trades,
        }

    adversarial_args = dict(attack=attacks[args.tr_attack],
                            attack_args=dict(net=model,
                                             data_params=data_params,
                                             attack_params=attack_params,
                                             verbose=False))

    #--------------------------------------------------#
    #------------------ Train-Test-Attack -------------#
    #--------------------------------------------------#

    NN = NeuralNetwork(model, train_loader, test_loader, optimizer, scheduler)

    if args.train:
        NN.train_model(logger, epoch_type=args.tr_epoch_type, num_epochs=args.epochs,
                       log_interval=args.log_interval, adversarial_args=adversarial_args)

        if not os.path.exists(args.directory + "checkpoints/frontends/"):
            os.makedirs(args.directory + "checkpoints/frontends/")
        NN.save_model(checkpoint_dir=args.directory + "checkpoints/frontends/" + checkpoint_name)

    else:
        NN.load_model(checkpoint_dir=args.directory + "checkpoints/frontends/" + checkpoint_name)
        logger.info("Clean test accuracy")
        test_loss, test_acc = NN.eval_model()
        logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    # if args.analyze_network:
    #     loss_landscape(model=model, data_loader=test_loader, img_index=0)
    #     outputs = frontend_analysis(model=frontend, test_loader=test_loader)
    #     breakpoint()

    if args.black_box:
        attack_loader = imagenette_black_box(args)
        test_loss, test_acc = NN.eval_model_blackbox(attack_loader=attack_loader)
        logger.info("Black Box test accuracy")
        logger.info(f'Blackbox Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

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
            adversarial_args=adversarial_args, progress_bar=True, save_blackbox=False)
        logger.info(f'{args.attack} test \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}\n')


if __name__ == "__main__":
    main()
