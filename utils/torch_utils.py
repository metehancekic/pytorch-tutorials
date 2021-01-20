import numpy as np

import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from ..models.custom_layers import TReLU


def neural_network_pruner(model):

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=0.6)
            prune.remove(module, "weight")
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=0.5)
            prune.remove(module, "weight")


def neural_network_sparsity(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            print("Sparsity in {:}.weight: {:.2f}%".format(module, 100. *
                                                           float(torch.sum(module.weight == 0)) / float(module.weight.nelement())))
        elif isinstance(module, torch.nn.Linear):
            print("Sparsity in {:}.weight: {:.2f}%".format(module, 100. *
                                                           float(torch.sum(module.weight == 0)) / float(module.weight.nelement())))


class Hook():
    def __init__(self, module, backward=False):
        if backward == False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output.detach().cpu().numpy()

    def close(self):
        self.hook.remove()


def intermediate_layer_outputs(args, data_params, model, data_loader, device):
    from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT
    from deepillusion.torchattacks.analysis import whitebox_test
    from deepillusion.torchattacks.analysis.plot import loss_landscape
    # if isinstance(module, torch.nn.Conv2d):
    # register hooks on each layer

    for name, module in model.named_modules():
        # breakpoint()
        if isinstance(module, TReLU):
            hookF = Hook(module)

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

    adversarial_args = dict(attack=attacks[args.attack],
                            attack_args=dict(net=model,
                                             data_params=data_params,
                                             attack_params=attack_params,
                                             verbose=False))

    # frontend_output = []
    # frontend_output_adv = []
    activation_list = []
    activation_list_adv = []
    images = []
    images_adv = []
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        out = model(X)

        adversarial_args["attack_args"]["net"] = model
        adversarial_args["attack_args"]["x"] = X
        adversarial_args["attack_args"]["y_true"] = y
        perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])

        images.append(X.detach().cpu().numpy())

        X += perturbs

        images_adv.append(X.detach().cpu().numpy())

        # frontend_output.append(activation['frontend'])
        activation_list.append(hookF.output)

        out = model(X)

        activation_list_adv.append(hookF.output)
        # frontend_output_adv.append(activation['frontend'])

    images = np.concatenate(tuple(images))
    images_adv = np.concatenate(tuple(images_adv))
    # frontend_output = np.concatenate(tuple(frontend_output))
    # frontend_output_adv = np.concatenate(tuple(frontend_output_adv))
    activation_list = np.concatenate(tuple(activation_list))
    activation_list_adv = np.concatenate(tuple(activation_list_adv))

    return images, images_adv, activation_list, activation_list_adv
