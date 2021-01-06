import torch


def intermediate_activations(model, data_loader, device):
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

    activation_list = []
    for X, y in data_loader:
        out = model(X.to(device))
        activation_list.append(activation['bn1'])

    return activation_list
