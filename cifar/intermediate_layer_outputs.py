import torch
from deepillusion.torchattacks import FGSM, RFGSM, PGD, PGD_EOT
from deepillusion.torchattacks.analysis import whitebox_test
from deepillusion.torchattacks.analysis.plot import loss_landscape


def intermediate_activations(args, data_params, model, data_loader, device):
    activation = {}

    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()
        return hook

    h1 = model.encoder.lp.register_forward_hook(getActivation('frontend'))
    h2 = model.decoder.features[0].register_forward_hook(getActivation('conv1'))
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

    frontend_list = []
    frontend_list_adv = []
    activation_list = []
    activation_list_adv = []
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        out = model(X)

        adversarial_args["attack_args"]["net"] = model
        adversarial_args["attack_args"]["x"] = X
        adversarial_args["attack_args"]["y_true"] = y
        perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
        X += perturbs

        frontend_list.append(activation['frontend'])
        activation_list.append(activation['conv1'])

        out = model(X)

        activation_list_adv.append(activation['conv1'])
        frontend_list_adv.append(activation['frontend'])

    return frontend_list, frontend_list_adv, activation_list, activation_list_adv
