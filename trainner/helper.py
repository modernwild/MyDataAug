import copy
import imp
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


def get_sgd(params, conf):
    return torch.optim.SGD(params, lr=conf.lr, momentum=conf.momentum, weight_decay=conf.weight_decay, nesterov=True)


def get_adam(params, conf):
    # return torch.optim.Adam(params, lr=conf.lr, weight_decay=conf.weight_decay)
    return torch.optim.Adam(params, lr=conf.lr)


# criterion
def get_criterion(conf):
    reduction = 'mean'
    if 'reduction' in conf:
        reduction = conf.reduction
    return eval('nn.' + conf.criterion)(reduction=reduction).cuda()


def get_params(model, conf=None):
    if conf is not None and 'prams_group' in conf:
        prams_group = conf.prams_group
        lr_group = conf.lr_group
        params = []
        for pram, lr in zip(prams_group, lr_group):
            params.append({'params': model.module.get_params(pram), 'lr': lr})
        return params
    return model.parameters()


def get_train_setting(model, conf):
    # optimizer
    if conf.lr_optimize:
        extractor_params = model.get_params(prefix='extractor')
        classifier_params = model.get_params(prefix='classifier')
        lr_cls = conf.lr
        lr_extractor = 0.1 * lr_cls
        if 'lr_rate' in conf:
            lr_extractor = conf.lr_rate * lr_cls
        params = [
            {'params': classifier_params, 'lr': lr_cls},
            {'params': extractor_params, 'lr': lr_extractor}
        ]
    else:
        params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=conf.lr, momentum=conf.momentum, weight_decay=conf.weight_decay)
    if conf.step.enable:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=conf.step.size, gamma=conf.step.gamma)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # criterion
    criterion = nn.CrossEntropyLoss().cuda()

    return criterion, optimizer, scheduler
