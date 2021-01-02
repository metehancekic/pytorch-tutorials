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

from .trades import trades_loss

__all__ = ['adversarial_epoch', 'adversarial_test']


class NeuralNetwork(object):

    def __init__(self, model, train_loader, test_loader, optimizer, scheduler=None):
        super(NeuralNetwork, self).__init__()

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_model(logger, single_epoch=None, num_epochs=100, log_interval=2, adversarial_args=None):
        if single_epoch is None:
            single_epoch = self.adversarial_epoch
        logger.info("Standard training")
        logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')

        epoch_args = dict(model=self.model,
                          train_loader=self.train_loader,
                          optimizer=self.optimizer,
                          scheduler=self.scheduler,
                          adversarial_args=adversarial_args)
        test_args = dict(model=self.model,
                         test_loader=self.test_loader)

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            train_loss, train_acc = single_epoch(**epoch_args)
            end_time = time.time()
            lr = scheduler.get_lr()[0]
            logger.info(f'{epoch} \t {end_time - start_time:.0f} \t \t {lr:.4f} \t {train_loss:.4f} \t {train_acc:.4f}')
            if epoch % log_interval == 0 or epoch == num_epochs:
                test_loss, test_acc = adversarial_test(**test_args)
                logger.info(f'Test  \t loss: {test_loss:.4f} \t acc: {test_acc:.4f}')

    def save_model(checkpoint_dir):
        torch.save(self.model.state_dict(), checkpoint_dir)

    def load_model(checkpoint_dir):
        self.model.load_state_dict(torch.load(checkpoint_dir))

    def eval_model(progress_bar=False, adversarial_args=None, save_blackbox=False):
        """
        Description: Evaluate model with test dataset,
            if adversarial args are present then adversarially perturbed test set.
        Input :
            adversarial_args :                   (dict)
                attack:                          (deepillusion.torchattacks)
                attack_args:                     (dict)
                    attack arguments for given attack except "x" and "y_true"
            progress_bar: Progress bar           (Bool)
        Output:
            train_loss : Train loss              (float)
            train_accuracy : Train accuracy      (float)
        """

        device = self.model.parameters().__next__().device

        self.model.eval()

        perturbed_data = []
        perturbed_labels = []
        test_loss = 0
        test_correct = 0
        if progress_bar:
            iter_test_loader = tqdm(
                iterable=self.test_loader,
                unit="batch",
                leave=False)
        else:
            iter_test_loader = self.test_loader

        for data, target in iter_test_loader:
            data, target = data.to(device), target.to(device)

            # Adversary
            if adversarial_args and adversarial_args["attack"]:
                adversarial_args["attack_args"]["net"] = model
                adversarial_args["attack_args"]["x"] = data
                adversarial_args["attack_args"]["y_true"] = target
                perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
                data += perturbs

            output = self.model(data)

            if save_blackbox:
                perturbed_data.append(data.detach().cpu().numpy())
                perturbed_labels.append(target.detach().cpu().numpy())

            cross_ent = nn.CrossEntropyLoss()
            test_loss += cross_ent(output, target).item() * data.size(0)

            pred = output.argmax(dim=1, keepdim=False)
            test_correct += pred.eq(target.view_as(pred)).sum().item()

        if save_blackbox:
            perturbed_data = np.concatenate(tuple(perturbed_data))
            perturbed_labels = np.concatenate(tuple(perturbed_labels))

        test_size = len(self.test_loader.dataset)

        return test_loss/test_size, test_correct/test_size

    def adversarial_epoch(adversarial_args=None):

        self.model.train()
        device = self.model.parameters().__next__().device

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

            self.optimizer.zero_grad()
            output = self.model(data)
            cross_ent = nn.CrossEntropyLoss()
            loss = cross_ent(output, target)
            loss.backward()
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            train_loss += loss.item() * data.size(0)
            pred_adv = output.argmax(dim=1, keepdim=False)
            train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

        train_size = len(self.train_loader.dataset)

        return train_loss/train_size, train_correct/train_size


def cosine_epoch(model, train_loader, optimizer, scheduler=None, adversarial_args=None):
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
            data_adv = data + perturbs

        optimizer.zero_grad()
        output, embedding = model(data)
        output_adv, embedding_adv = model(data_adv)
        cross_ent = nn.CrossEntropyLoss()
        cos = nn.CosineSimilarity()
        loss = cross_ent(output, target) + 1000 * (1 - cos(embedding, embedding_adv).mean())
        # breakpoint()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=False)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()
        # breakpoint()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def trades_epoch(model, train_loader, optimizer, scheduler=None, adversarial_args=None):

    model.train()

    device = model.parameters().__next__().device

    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # calculate robust loss
        loss, output = trades_loss(model=model,
                                   x_natural=data,
                                   y=target,
                                   optimizer=optimizer,
                                   step_size=adversarial_args["attack_args"]["attack_params"]["step_size"],
                                   epsilon=adversarial_args["attack_args"]["attack_params"]["eps"],
                                   perturb_steps=adversarial_args["attack_args"]["attack_params"]["num_steps"],
                                   beta=adversarial_args["attack_args"]["attack_params"]["beta"],
                                   distance='l_inf')
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        train_loss += loss.item() * data.size(0)
        pred_adv = output.argmax(dim=1, keepdim=False)
        train_correct += pred_adv.eq(target.view_as(pred_adv)).sum().item()

    train_size = len(train_loader.dataset)

    return train_loss/train_size, train_correct/train_size


def reconstruction_epoch(model, train_loader, optimizer, scheduler=None):
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
    with tqdm(
        total=len(train_loader),
        unit="Bt",
        unit_scale=True,
        unit_divisor=1000,
        leave=False,
        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
            ) as pbar:
        for batch_idx, (images, _) in enumerate(train_loader):

            if isinstance(images, list):
                images = images[0]

            images = images.to(device)

            optimizer.zero_grad()
            output = model(images)
            criterion = nn.MSELoss()

            loss = criterion(output, images)
            loss.backward()
            optimizer.step()
            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
                scheduler.step()

            nb_img_so_far = (batch_idx + 1) * train_loader.batch_size

            train_loss += loss.item() * train_loader.batch_size
            pbar.set_postfix(
                Train_Loss=train_loss / nb_img_so_far, refresh=True,
                )
            pbar.update(1)

    if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
        scheduler.step()

    train_size = len(train_loader) * train_loader.batch_size

    return train_loss/train_size


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

        output = model(data)

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()
    # print(test_correct)

    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size


def save_blackbox(model, test_loader, adversarial_args=None, verbose=False, progress_bar=False):
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

    perturbed_data = []
    perturbed_labels = []
    for data, target in iter_test_loader:

        data, target = data.to(device), target.to(device)

        # Adversary
        if adversarial_args and adversarial_args["attack"]:
            adversarial_args["attack_args"]["net"] = model
            adversarial_args["attack_args"]["x"] = data
            adversarial_args["attack_args"]["y_true"] = target
            perturbs = adversarial_args['attack'](**adversarial_args["attack_args"])
            data += perturbs

        output = model(data)

        perturbed_data.append(data.detach().cpu().numpy())
        perturbed_labels.append(target.detach().cpu().numpy())

        cross_ent = nn.CrossEntropyLoss()
        test_loss += cross_ent(output, target).item() * data.size(0)

        pred = output.argmax(dim=1, keepdim=False)
        test_correct += pred.eq(target.view_as(pred)).sum().item()

    perturbed_data = np.concatenate(tuple(perturbed_data))
    perturbed_labels = np.concatenate(tuple(perturbed_labels))

    np.savez("/home/metehan/pytorch-tutorials/imagenette/data/black_box_resnet",
             perturbed_data, perturbed_labels)
    test_size = len(test_loader.dataset)

    return test_loss/test_size, test_correct/test_size


def reconstruction_test(model, test_loader):

    model.eval()

    test_loss = 0

    device = model.parameters().__next__().device

    for batch_idx, (images, _) in enumerate(test_loader):

        if isinstance(images, list):
            images = images[0]

        images = images.to(device)

        output = model(images)
        criterion = nn.MSELoss()
        loss = criterion(output, images)

        test_loss += loss.item() * test_loader.batch_size
        nb_img_so_far = (batch_idx + 1) * test_loader.batch_size

    test_size = len(test_loader) * test_loader.batch_size

    return test_loss / test_size


def frontend_outputs(model, test_loader):

    model.eval()

    device = model.parameters().__next__().device

    orig_images = []
    before_drelu = []
    after_drelu = []
    after_frontend = []
    for batch_idx, (images, _) in enumerate(test_loader):

        if isinstance(images, list):
            images = images[0]

        images = images.to(device)

        o1, o2 = model.firstlayer(images)
        o3 = model(images)
        orig_images.append(images.detach().cpu().numpy())
        before_drelu.append(o1.detach().cpu().numpy())
        after_drelu.append(o2.detach().cpu().numpy())
        after_frontend.append(o3.detach().cpu().numpy())

    orig_images = np.concatenate(tuple(orig_images))
    before_drelu = np.concatenate(tuple(before_drelu))
    after_drelu = np.concatenate(tuple(after_drelu))
    after_frontend = np.concatenate(tuple(after_frontend))

    return orig_images, before_drelu, after_drelu, after_frontend


def frontend_analysis(model, test_loader):

    model.eval()

    device = model.parameters().__next__().device

    orig_images = []
    o1 = []
    o2 = []
    o3 = []
    o4 = []
    o5 = []
    for batch_idx, (images, _) in enumerate(test_loader):

        if isinstance(images, list):
            images = images[0]

        images = images.to(device)

        outputs = model.analysis(images)
        orig_images.append(images.detach().cpu().numpy())
        o1.append(outputs[0].detach().cpu().numpy())
        o2.append(outputs[1].detach().cpu().numpy())
        o3.append(outputs[2].detach().cpu().numpy())
        o4.append(outputs[3].detach().cpu().numpy())
        o5.append(outputs[4].detach().cpu().numpy())

    orig_images = np.concatenate(tuple(orig_images))
    o1 = np.concatenate(tuple(o1))
    o2 = np.concatenate(tuple(o2))
    o3 = np.concatenate(tuple(o3))
    o4 = np.concatenate(tuple(o4))
    o5 = np.concatenate(tuple(o5))

    return [orig_images, o1, o2, o3, o4, o5]
