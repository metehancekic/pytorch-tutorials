import torch
from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT
from deepillusion.torchattacks.analysis import whitebox_test
from deepillusion.torchattacks.analysis.plot import loss_landscape


def intermediate_activations(args, data_params, model, data_loader, device):
    activation = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    h1 = model.block3[0].bn1.register_forward_hook(getActivation('bn1'))
    # h2 = model.block3[1].bn1.register_forward_hook(getActivation('bn2'))
    # h3 = model.block3[2].bn1.register_forward_hook(getActivation('bn3'))
    # h4 = model.block3[3].bn1.register_forward_hook(getActivation('bn4'))
    # h5 = model.block3[4].bn1.register_forward_hook(getActivation('bn5'))

    # h1 = model.block3[0].bn2.register_forward_hook(getActivation('bn6'))
    # h1 = model.block3[1].bn2.register_forward_hook(getActivation('bn7'))
    # h1 = model.block3[2].bn2.register_forward_hook(getActivation('bn8'))
    # h1 = model.block3[3].bn2.register_forward_hook(getActivation('bn9'))
    # h1 = model.block3[4].bn2.register_forward_hook(getActivation('bn10'))

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

    activation_list = []
    activation_list_adv = []
    for X, y in data_loader:
        X = X.to(device)
        out = model(X)

        adversarial_args["attack_args"]["net"] = model
        adversarial_args["attack_args"]["x"] = X
        adversarial_args["attack_args"]["y_true"] = y
        perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
        X += perturbs

        activation_list.append(activation['bn1'])

        out = model(X)

        activation_list_adv.append(activation['bn1'])

    return activation_list, activation_list_adv
