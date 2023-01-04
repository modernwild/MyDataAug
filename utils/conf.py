import argparse
import os
import random
import numpy as np
import torch
import yaml
import copy
from easydict import EasyDict as edict
from torch.backends import cudnn

parser = argparse.ArgumentParser(description='Evaluate GG-CNN')
parser.add_argument('--config_file', default='config/comm.yml', type=str, help='Visualise the network output')
parser.add_argument('--iteal', type=int, help='Visualise the network output')
parser.add_argument('--evaluate', action='store_true', help='Visualise the network output')
# Dataset & Data & Training
parser.add_argument('--description', type=str, help='Visualise the network output')
parser.add_argument('--dataset_name', type=str, help='Dataset Name ("cornell" or "jaquard")')
parser.add_argument('--dataset_path', type=str, help='Path to dataset')
parser.add_argument('--resume', type=str, help='Path to dataset')

parser.add_argument('--epochs', type=int, help='Visualise the network output')
parser.add_argument('--batch_size', type=int, help='Visualise the network output')
parser.add_argument('--lr', type=float, help='Visualise the network output')
parser.add_argument('--cropsize', type=int, help='Visualise the network output')

parser.add_argument('--logdir', type=str, help='Visualise the network output')
parser.add_argument('--num_workers', type=int, help='Visualise the network output')
args = parser.parse_args()


# parser.add_argument('--config', default='config/comm.yml', type=str, help='config files')
# ----------------------------------------------------------------------------------------
# base


def str2bool(v):
    return v.lower() in ('true')


def get_config_from_parser():
    """
    将控制台输出的参数转化为edict
    :return: 控制台参数
    """
    config, unparsed = parser.parse_known_args()
    cfg = edict(config.__dict__)
    return edict(cfg)


def get_conf_from_file(file_path):
    """
    读取配置文件的参数
    :param file_path: 配置文件路径
    :return: 配置文件参数
    """
    filename = file_path
    with open(filename, 'r', encoding='UTF-8') as f:
        return edict(yaml.safe_load(f))


def get_config():
    """
    合并parser参数（高优先级）和配置文件的参数，转为一个dic
    :return:
    """
    # 读取控制台参数
    pars1 = get_config_from_parser()
    # 读取配置文件参数
    pars2 = get_conf_from_file(pars1['config_file'])

    # 合并两者配置，取pars2为高优先级
    for k, v in pars1.items():
        if v is not None:
            pars2[k] = v
    return pars2


def print_conf(opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        # default = self.parser.get_default(k)
        # if v != default:
        #     comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    return message


def set_env(cfg):
    cfg["seed"] = 123
    # set seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)  # cpu vars
    torch.manual_seed(cfg.seed)  # cpu  vars
    torch.cuda.manual_seed(cfg.seed)  # cpu  vars
    torch.cuda.manual_seed_all(cfg.seed)  # gpu vars
    if 'cudnn' in cfg:
        torch.backends.cudnn.benchmark = cfg.cudnn
    else:
        torch.backends.cudnn.benchmark = False

    # cudnn.deterministic = True
    # os.environ["NUMEXPR_MAX_THREADS"] = '16'
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
