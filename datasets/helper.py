import imp
import os

import logging as logger
import torch.utils.data as data
import datasets.cub_data as cub
from importlib import import_module


def get_dataset(conf):
    module_ = import_module('datasets.{}_data'.format(conf.dataset_name))
    get_dataset_fun = getattr(module_, "get_dataset")

    ds_train, ds_test = get_dataset_fun(conf)
    return ds_train, ds_test


def get_dataloader(conf):
    train_dataset, valid_dataset = get_dataset(conf)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        shuffle=True,
        pin_memory=True
    )
    val_loader = data.DataLoader(
        valid_dataset,
        batch_size=conf.batch_size,
        num_workers=conf.num_workers,
        pin_memory=True
    )
    logger.info('Training batchs: {}'.format(len(train_loader)))
    logger.info('Validation batchs: {}'.format(len(val_loader)))

    return train_loader, val_loader
