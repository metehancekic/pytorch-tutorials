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
from ..utils import mean_l1_norm, mean_l2_norm, mean_linf_norm
# from .gabor_trial import plot_image


def main():

    logger, args, device, train_loader, test_loader, data_params = initialize_everything()

    #--------------------------------------------------#
    #---------- Set up Model and checkpoint name ------#
    #--------------------------------------------------#
    from ..models import ResNet, VGG, VGG_modified, VGG_modified2

    checkpoint_name = args.model
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

    NN = NeuralNetwork(model, args.model, optimizer, scheduler)

    if args.train:
        NN.train_model(train_loader, test_loader, logger, epoch_type=args.tr_epoch_type, num_epochs=args.epochs,
                       log_interval=args.log_interval, adversarial_args=adversarial_args)
        if args.prune:
            from ..utils import neural_network_pruner
            neural_network_pruner(model)
            checkpoint += "_pruned"
        if not os.path.exists(args.directory + "checkpoints/frontends/"):
            os.makedirs(args.directory + "checkpoints/frontends/")
        NN.save_model(checkpoint_dir=args.directory +
                      "checkpoints/frontends/" + checkpoint_name + ".pt")

    else:
        if args.prune:
            checkpoint += "_pruned"
        NN.load_model(checkpoint_dir=args.directory +
                      "checkpoints/frontends/" + checkpoint_name + ".pt")
        logger.info("Clean test accuracy")
        test_loss, test_acc = NN.eval_model(test_loader)
        logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')
        from ..utils import neural_network_sparsity
        # neural_network_pruner(model)
        neural_network_sparsity(model)
        # logger.info("Clean test accuracy")
        test_loss, test_acc = NN.eval_model(test_loader)
        # logger.info("After pruning")
        # logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')
    if args.analyze_network:
        # loss_landscape(model=model, data_loader=test_loader, img_index=0)
        # outputs = frontend_analysis(model=frontend, test_loader=test_loader)
        from ..utils import intermediate_layer_outputs
        imgs, imgs_adv, act, act_adv = intermediate_layer_outputs(
            args, data_params, model, test_loader, device)

        l1_imgs = mean_l1_norm(imgs)
        l2_imgs = mean_l2_norm(imgs)
        linf_imgs = mean_linf_norm(imgs)

        l1_imgs_diff = mean_l1_norm(imgs_adv-imgs)
        l2_imgs_diff = mean_l2_norm(imgs_adv-imgs)
        linf_imgs_diff = mean_linf_norm(imgs_adv-imgs)

        l1_act = mean_l1_norm(act)
        l2_act = mean_l2_norm(act)
        linf_act = mean_linf_norm(act)

        l1_act_diff = mean_l1_norm(act_adv-act)
        l2_act_diff = mean_l2_norm(act_adv-act)
        linf_act_diff = mean_linf_norm(act_adv-act)

        print(f"Mean L1 norm of images:  {l1_imgs}")
        print(f"Mean L2 norm of images:  {l2_imgs}")
        print(f"Mean Li norm of images:  {linf_imgs}")

        print(f"Mean L1 norm of error_img:  {l1_imgs_diff}")
        print(f"Mean L2 norm of error_img:  {l2_imgs_diff}")
        print(f"Mean Li norm of error_img:  {linf_imgs_diff}")

        print(f"Mean L1 norm of first conv output:  {l1_act}")
        print(f"Mean L2 norm of first conv output:  {l2_act}")
        print(f"Mean Li norm of first conv output:  {linf_act}")

        print(f"Mean L1 norm of error_conv:  {l1_act_diff}")
        print(f"Mean L2 norm of error_conv:  {l2_act_diff}")
        print(f"Mean Li norm of error_conv:  {linf_act_diff}")

        from matplotlib import pyplot as plt
        bins = np.arange(0, 1, 0.01)
        plt.figure()
        plt.hist(act[0].reshape(-1), bins)
        plt.savefig(args.directory + "figs/" + "first_layer_activation_hist_1")

        plt.figure()
        plt.hist(act_adv[0].reshape(-1) - act[0].reshape(-1), bins)
        plt.savefig(args.directory + "figs/" + "first_layer_activation_difference_hist_1")

        plt.figure()
        plt.hist(act.reshape(-1), bins)
        plt.savefig(args.directory + "figs/" + "first_layer_activation_hist")

        plt.figure()
        plt.hist(act_adv.reshape(-1) - act.reshape(-1), bins)
        plt.savefig(args.directory + "figs/" + "first_layer_activation_difference_hist")

        print("Sparstiy level for clean input : " + str(np.count_nonzero(act)/np.size(act)))
        print("Sparstiy level for adversarial input : " +
              str(np.count_nonzero(act_adv)/np.size(act_adv)))

        breakpoint()

    if args.black_box:
        attack_loader = cifar10_black_box(args)
        test_loss, test_acc = NN.eval_model(attack_loader)
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
            test_loader, adversarial_args=adversarial_args, progress_bar=True, save_blackbox=False)
        logger.info(f'{args.attack} test \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}\n')


if __name__ == "__main__":
    main()
