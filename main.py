import logging
import os

import pandas as pd
import torch
import torch.utils.data

import logging as logger
import utils
import datasets.helper
import networks.helper
import utils.draw
import trainner
from trainner.test import valid_one_epoch
from trainner.train import train_one_epoch
from utils.conf import get_config
from utils.io import set_logger, load_checkpoint, save_checkpoint

import trainner.helper


def _print_multi_res(res):
    str = ''
    for k, v in res.items(): str = str + f"{k}:{v.value():.2f}"
    logger.info(str)


def main(config):
    train_loader, val_loader = datasets.helper.get_dataloader(config)

    device = torch.device("cuda:0")
    network = networks.helper.get_model(config)
    network.to(device)

    criterion, optimizer, scheduler = trainner.helper.get_train_setting(network, config)

    best_score = 0
    epoch_start = 0
    if 'resume' in config:  # 读取检查点
        checkpoint_dict = utils.io.load_checkpoint(network, config.resume)
        epoch_start = checkpoint_dict['epoch']
        print(f'Resuming training process from epoch {epoch_start}...')
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])

    if config.evaluate:
        res = trainner.valid_one_epoch(network, device, val_loader, criterion, config)
        print("score:{} ".format(res['score'].avg))
        return

    # ------main loop-----
    history = {'epoch': [],
               'train_loss': [],
               'valid_acc': [],
               'best_acc': []}

    eval_epoch = 0 if 'eval_epoch' not in config or config.eval_epoch is None else config.eval_epoch
    is_best = False
    for epoch in range(epoch_start, config.epochs):

        # =====training=======
        train_res = trainner.train_one_epoch(network, device, train_loader, optimizer, criterion, config)
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch:\t{}\ttrain'.format(epoch + 1))
        _print_multi_res(train_res)
        if config.scheduler_enable:
            scheduler.step()

        if epoch >= eval_epoch:
            # =====validating=====
            val_res = trainner.valid_one_epoch(network, device, val_loader, criterion, config)
            is_best = val_res['score'].value() > best_score
            best_score = max(val_res['score'].value(), best_score)

            logging.info('Epoch:\t{}\ttest\tbest_acc: {}'.format(epoch + 1, best_score))
            _print_multi_res(val_res)
        save_checkpoint(
            {'epoch': epoch + 1,
             'state_dict': network.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(),
             'best_score': best_score
             }, is_best, outdir=config.outdir, iteral=config.iteal)

        # ==== save to csv =====
        history['epoch'].append(epoch)
        history['train_loss'].append(train_res['loss'].value())
        if epoch >= eval_epoch:
            history['valid_acc'].append(val_res['score'].value())
        else:
            history['valid_acc'].append(0)
        history['best_acc'].append(best_score)
    # save log
    torch.save(network, os.path.join(config.outdir, 'final_model.pth'))
    df = pd.DataFrame(history)
    history_path = os.path.join(config.outdir, 'history.csv')
    df.to_csv(history_path, index=False)
    utils.draw.draw_loss(history_path, save_path=os.path.join(config.outdir, 'history.png'))


if __name__ == '__main__':
    conf = utils.conf.get_config()
    utils.conf.set_env(conf)  # 设置seed和并行参数
    utils.io.set_outdir(conf)  # 创建输出文件夹

    utils.io.set_logger(conf)  # 设置logger
    main(conf)
