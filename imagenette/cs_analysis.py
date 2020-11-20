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
from ..models.custom_layers import CenterSurroundModule, AutoEncoder, Decoder, CenterSurroundConv
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
            logging.FileHandler(args.directory + 'logs/' + "CenterSurroundModule" + '.log'),
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

    attack_loader = imagenette_black_box(args)
    cs_frontend = CenterSurroundModule(beta=args.beta).to(device)
    cs_fe = CenterSurroundConv(beta=args.beta).to(device)
    CNN = ResNet().to(device)
    model = AutoEncoder(cs_fe, CNN).to(device)
    if device == "cuda":
        cs_frontend = torch.nn.DataParallel(cs_frontend)
        cudnn.benchmark = True

    # for beta in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
    #     cs_frontend = CenterSurroundModule(beta=beta).to(device)
    #     if device == "cuda":
    #         cs_frontend = torch.nn.DataParallel(cs_frontend)
    #         cudnn.benchmark = True

    #     logger.info(cs_frontend)
    #     logger.info("\n")

    #     images, labels = test_loader.__iter__().next()
    #     images = images.to(device)
    #     reconstructions = cs_frontend(images)
    #     import matplotlib.pyplot as plt
    #     for i in range(10):
    #         plt.figure()
    #         plt.imshow(reconstructions.permute(0, 2, 3, 1).detach().cpu().numpy()[i])
    #         plt.savefig(args.directory + "figures/" + f"cs_b_{int(beta)}_i_{i}.pdf")

    cs_frontend.eval()

    attacks = dict(Standard=None,
                   PGD=PGD,
                   FGSM=FGSM,
                   RFGSM=RFGSM,
                   PGD_EOT=PGD_EOT)
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

    data_params = {"x_min": 0., "x_max": 1.}

    adversarial_args = dict(attack=attacks[args.attack],
                            attack_args=dict(net=model,
                                             data_params=data_params,
                                             attack_params=attack_params,
                                             loss_function="cross_entropy",
                                             verbose=False,
                                             progress_bar=True))

    progress_bar = True
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=attack_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = attack_loader

    clean_out_before_drelu = []
    clean_out_after_drelu = []
    adversarial_out_before_drelu = []
    adversarial_out_after_drelu = []
    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        adversarial_args["attack_args"]["net"] = model
        adversarial_args["attack_args"]["x"] = data
        adversarial_args["attack_args"]["y_true"] = target
        perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
        data_adv = data + perturbs

        clean_before_drelu, clean_after_drelu = cs_frontend.firstlayer(data)
        adversarial_before_drelu, adversarial_after_drelu = cs_frontend.firstlayer(data_adv)
        clean_out_before_drelu.append(clean_before_drelu.detach().cpu().numpy())
        clean_out_after_drelu.append(clean_after_drelu.detach().cpu().numpy())
        adversarial_out_before_drelu.append(adversarial_before_drelu.detach().cpu().numpy())
        adversarial_out_after_drelu.append(adversarial_after_drelu.detach().cpu().numpy())

    clean_out_before_drelu = np.concatenate(tuple(clean_out_before_drelu))
    clean_out_after_drelu = np.concatenate(tuple(clean_out_after_drelu))
    adversarial_out_before_drelu = np.concatenate(tuple(adversarial_out_before_drelu))
    adversarial_out_after_drelu = np.concatenate(tuple(adversarial_out_after_drelu))
    # breakpoint()
    # if progress_bar:
    #     iter_test_loader = tqdm(
    #         iterable=test_loader,
    #         unit="batch",
    #         leave=False)
    # else:
    #     iter_test_loader = test_loader

    # clean_out = []
    # for data, target in iter_test_loader:

    #     data, target = data.to(device), target.to(device)

    #     output = cs_frontend(data)
    #     clean_out.append(output.detach().cpu().numpy())

    # clean_out = np.concatenate(tuple(clean_out))

    import matplotlib.pyplot as plt
    image_folder = args.directory + "figures/"
    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

    for i in range(6):
        plt.figure()
        plt.hist(clean_out_before_drelu[:, i, :, :].ravel(), bins=256, range=(
            clean_out_before_drelu[:, i, :, :].min(), clean_out_before_drelu[:, i, :, :].max()), fc='k', ec='k', density=True)
        plt.savefig(image_folder + f"hist_clean_before_drelu_{i}.pdf")

        plt.figure()
        plt.hist(clean_out_after_drelu[:, i, :, :].ravel(), bins=256, range=(
            clean_out_after_drelu[:, i, :, :].min(), clean_out_after_drelu[:, i, :, :].max()), fc='k', ec='k', density=True)
        plt.savefig(image_folder + f"hist_clean_after_drelu_{i}.pdf")

        plt.figure()
        plt.hist(adversarial_out_before_drelu[:, i, :, :].ravel(), bins=256, range=(
            adversarial_out_before_drelu[:, i, :, :].min(), adversarial_out_before_drelu[:, i, :, :].max()), fc='k', ec='k', density=True)
        plt.savefig(image_folder + f"hist_adversarial_before_drelu_{i}.pdf")

        plt.figure()
        plt.hist(adversarial_out_after_drelu[:, i, :, :].ravel(), bins=256, range=(
            adversarial_out_after_drelu[:, i, :, :].min(), adversarial_out_after_drelu[:, i, :, :].max()), fc='k', ec='k', density=True)
        plt.savefig(image_folder + f"hist_adversarial_after_drelu_{i}.pdf")

    difference_after = adversarial_out_after_drelu - clean_out_after_drelu
    for i in range(6):
        plt.figure()
        plt.hist(difference_after[:, i, :, :].ravel(), bins=256, range=(
            difference_after[:, i, :, :].min(), difference_after[:, i, :, :].max()), fc='k', ec='k', density=True)
        plt.savefig(image_folder + f"hist_difference_after_{i}.pdf")

    breakpoint()


    # o_images, b_drelu, a_drelu, a_frontend = frontend_outputs(cs_frontend, test_loader)
    # import matplotlib.pyplot as plt
    # for i in range(3):
    #     plt.figure()
    #     plt.hist(o_images[:, i, :, :].ravel(), bins=256, range=(
    #         o_images[:, i, :, :].min(), o_images[:, i, :, :].max()), fc='k', ec='k')
    #     image_folder = args.directory + "figures/"
    #     if not os.path.exists(image_folder):
    #         os.mkdir(image_folder)
    #     plt.savefig(image_folder + f"hist_orig_images_{i}.pdf")
    #     plt.figure()
    #     plt.hist(a_frontend[:, i, :, :].ravel(), bins=256, range=(
    #         a_frontend[:, i, :, :].min(), a_frontend[:, i, :, :].max()), fc='k', ec='k')
    #     image_folder = args.directory + "figures/"
    #     if not os.path.exists(image_folder):
    #         os.mkdir(image_folder)
    #     plt.savefig(image_folder + f"hist_after_frontend_channel_{i}.pdf")
    # for i in range(6):
    #     plt.figure()
    #     plt.hist(b_drelu[:, i, :, :].ravel(), bins=256, range=(
    #         b_drelu[:, i, :, :].min(), b_drelu[:, i, :, :].max()), fc='k', ec='k')
    #     image_folder = args.directory + "figures/"
    #     if not os.path.exists(image_folder):
    #         os.mkdir(image_folder)
    #     plt.savefig(image_folder + f"hist_before_drelu_channel_{i}.pdf")
    #     plt.figure()
    #     plt.hist(b_drelu[:, i, :, :].ravel(), bins=256, range=(
    #         a_drelu[:, i, :, :].min(), a_drelu[:, i, :, :].max()), fc='k', ec='k')
    #     image_folder = args.directory + "figures/"
    #     if not os.path.exists(image_folder):
    #         os.mkdir(image_folder)
    #     plt.savefig(image_folder + f"hist_after_drelu_channel_{i}.pdf")
    # breakpoint()
    # Train network if args.train is set to True (You can set that true by calling '-tr' flag, default is False)
if __name__ == "__main__":
    main()
