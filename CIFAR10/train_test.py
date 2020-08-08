"""
Description: Training and testing functions for neural models

functions:
    train: Performs a single training epoch (if attack_args is present adversarial training)
    test: Evaluates model by computing accuracy (if attack_args is present adversarial testing)
"""

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['adversarial_epoch', 'adversarial_test']


def adversarial_epoch(model, train_loader, optimizer, scheduler=None, adversarial_args=None):
    """
    Description: Single epoch,
        if adversarial args are present then adversarial training.
    Input :
        model : Neural Network               (torch.nn.Module)
        train_loader : Data loader           (torch.utils.data.DataLoader)
        optimizer : Optimizer                (torch.nn.optimizer)
        scheduler: Scheduler (Optional)      (torch.optim.lr_scheduler.CyclicLR)
        adversarial_args :
            attack:                          (deepillusion.torchattacks)
            attack_args:
                attack arguments for given attack except "x" and "y_true"
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target in train_loader:

        data, target = data.to(device), target.to(device)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        optimizer.zero_grad()
        output = model(data)
        cross_ent = nn.CrossEntropyLoss()
        loss = cross_ent(output, target)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=True)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def randomized_epoch(model, train_loader, optimizer, scheduler=None, adversarial_args=None):
    """
    Description: Single epoch,
        if adversarial args are present then adversarial training.
    Input :
        model : Neural Network               (torch.nn.Module)
        train_loader : Data loader           (torch.utils.data.DataLoader)
        optimizer : Optimizer                (torch.nn.optimizer)
        scheduler: Scheduler (Optional)      (torch.optim.lr_scheduler.CyclicLR)
        adversarial_args :
            attack:                          (deepillusion.torchattacks)
            attack_args:
                attack arguments for given attack except "x" and "y_true"
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target in train_loader:

        data, target = data.to(device), target.to(device)
        batch_size = len(data)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        optimizer.zero_grad()

        cross_ent = nn.CrossEntropyLoss()

        output, _ = model(data)

        loss = cross_ent(output, target)
        loss.backward()
        optimizer.step()
        # if scheduler:
        #     scheduler.step()

        optimizer.zero_grad()

        rand_data = torch.randn(data.shape).to(device)
        _, pre_linear = model(rand_data)
        # breakpoint()
        l2_loss = 0.*torch.sum(torch.norm(pre_linear, dim=1))/batch_size
        l2_loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=True)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def with_random_inputs_epoch(model, train_loader, optimizer, scheduler=None, adversarial_args=None):
    """
    Description: Single epoch,
        if adversarial args are present then adversarial training.
    Input :
        model : Neural Network               (torch.nn.Module)
        train_loader : Data loader           (torch.utils.data.DataLoader)
        optimizer : Optimizer                (torch.nn.optimizer)
        scheduler: Scheduler (Optional)      (torch.optim.lr_scheduler.CyclicLR)
        adversarial_args :
            attack:                          (deepillusion.torchattacks)
            attack_args:
                attack arguments for given attack except "x" and "y_true"
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for data, target in train_loader:

        data, target = data.to(device), target.to(device)
        batch_size = len(data)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        optimizer.zero_grad()

        one_hot = torch.eye(10)
        one_hot_targets = one_hot[target].to(device)

        rand_data = torch.randn(data.shape).to(device)
        data = torch.cat((data, rand_data), dim=0)
        uniform_target = torch.ones_like(one_hot_targets)/10
        one_hot_targets = torch.cat((one_hot_targets, uniform_target), dim=0)

        output = model(data)
        log_probs = torch.nn.functional.log_softmax(output, dim=1)

        # breakpoint()

        # cross_ent = nn.CrossEntropyLoss()
        cross_ent = nn.KLDivLoss(reduction="batchmean")
        loss = cross_ent(log_probs, one_hot_targets)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output[:batch_size].argmax(dim=1, keepdim=True)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def adversarial_test_madhow(model, test_loader, adversarial_args=None, verbose=False, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        adversarial_args :                   (dict)
            attack:                          (deepillusion.torchattacks)
            attack_args:                     (dict)
                attack arguments for given attack except "x" and "y_true"
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    road = np.zeros((10, 2))
    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        output, indices = model(data)
        for i in range(1000):
            road[target[i], indices[i, target[i], 0]] += 1

        cross_ent = nn.CrossEntropyLoss()
        loss += cross_ent(output, target) * data.size(0)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)
    print(road)

    return test_loss/test_size, test_correct/test_size


def adversarial_test(model, test_loader, adversarial_args=None, verbose=False, progress_bar=False):
    """
    Description: Evaluate model with test dataset,
        if adversarial args are present then adversarially perturbed test set.
    Input :
        model : Neural Network               (torch.nn.Module)
        test_loader : Data loader            (torch.utils.data.DataLoader)
        adversarial_args :                   (dict)
            attack:                          (deepillusion.torchattacks)
            attack_args:                     (dict)
                attack arguments for given attack except "x" and "y_true"
        verbose: Verbosity                   (Bool)
        progress_bar: Progress bar           (Bool)
    Output:
        train_loss : Train loss              (float)
        train_accuracy : Train accuracy      (float)
    """

    device = model.parameters().__next__().device

    model.eval()

    test_loss = 0
    test_correct = 0
    if progress_bar:
        iter_test_loader = tqdm(
            iterable=test_loader,
            unit="batch",
            leave=False)
    else:
        iter_test_loader = test_loader

    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        output, _ = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=True)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size
