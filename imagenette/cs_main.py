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
from deepillusion.torchattacks import FGSM, RFGSM, PGD
from deepillusion.torchattacks.analysis import whitebox_test
# from deepillusion.torchdefenses import adversarial_epoch

# CIFAR10 TRAIN TEST CODES
from ..models import ResNet, VGG, MobileNet, MobileNetV2, PreActResNet, EfficientNet
from ..models.custom_layers import CenterSurroundModule, AutoEncoder, Decoder
from ..train_test import adversarial_epoch, adversarial_test, reconstruction_epoch, reconstruction_test, frontend_outputs
from ..read_datasets import imagenette
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
    cs_frontend = CenterSurroundModule(beta=).to(device)
    decoder = Decoder().to(device)
    model = AutoEncoder(cs_frontend, decoder).to(device)
    if device == "cuda":
        cs_frontend = torch.nn.DataParallel(cs_frontend)
        cudnn.benchmark = True

    logger.info(cs_frontend)
    logger.info("\n")

    # Which optimizer to be used for training
    optimizer = optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum,
                          weight_decay=args.weight_decay)

    lr_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=args.lr_min,
                                                  max_lr=args.lr_max, step_size_up=lr_steps/2,
                                                  step_size_down=lr_steps/2)
    # scheduler = None
    # Checkpoint Namer
    checkpoint_name = "CenterSurroundModule"
    # breakpoint()

    # images, labels = test_loader.__iter__().next()
    # images = images.to(device)
    # reconstructions = cs_frontend(images)
    # import matplotlib.pyplot as plt
    # for i in range(10):
    #     plt.figure()
    #     plt.imshow(reconstructions.permute(0, 2, 3, 1).detach().cpu().numpy()[i])
    #     plt.savefig(f"lap_low_{i}.pdf")

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
    with tqdm(
            total=args.epochs,
            initial=0,
            unit="ep",
            unit_scale=True,
            unit_divisor=1000,
            leave=True,
            bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
            ) as pbar:

        for epoch in range(1, args.epochs + 1):

            train_args = dict(model=model,
                              train_loader=train_loader,
                              optimizer=optimizer,
                              scheduler=scheduler)
            train_loss = reconstruction_epoch(**train_args)

            validation_args = dict(model=model,
                                   test_loader=test_loader)
            validation_loss = reconstruction_test(**validation_args)

            pbar.set_postfix(
                Val_Loss=f"{validation_loss:.4f}", refresh=True,
                )
            pbar.update(1)

        # Save model parameters
        if not os.path.exists(args.directory + "checkpoints/cs_frontends/"):
            os.makedirs(args.directory + "checkpoints/cs_frontends/")
        torch.save(model.state_dict(), args.directory +
                   "checkpoints/cs_frontends/" + checkpoint_name)

    # breakpoint()
    # images, labels = test_loader.__iter__().next()
    # images = images.to(device)
    # reconstructions = model(images)
    # import matplotlib.pyplot as plt
    # # plot_image(args, reconstructions.permute(0, 2, 3, 1).numpy()[
    # #            0], "recons.pdf", image_index=0, colored=True, show_image=False)
    # for i in range(10):
    #     plt.figure()
    #     plt.imshow(reconstructions.permute(0, 2, 3, 1).detach().cpu().numpy()[i])
    #     plt.savefig(f"recons_{i}.pdf")
    # breakpoint()


if __name__ == "__main__":
    main()
