import logging
import os
import shutil
import torch
from datetime import datetime
from core.conf import print_conf

def load_pretrained(model, conf, logger):
    model.load_state_dict(torch.load(conf.pretrain))
    logging.info('Load pretrain model from: {}'.format(conf.pretrain))


# ---------------load checkpoint--------------------
def load_checkpoint(model, pth_file):
    print('==> Reading from model checkpoint..')
    assert os.path.isfile(pth_file), 'Error: no model checkpoint directory found!'
    checkpoint = torch.load(pth_file)
    # args.start_epoch = checkpoint['epoch']
    # best_prec1 = checkpoint['best_prec1']

    pretrained_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)

    # model.module.load_state_dict(checkpoint['state_dict'])
    model.load_state_dict(model_dict)
    print("=> loaded model checkpoint '{}' (epoch {})".format(pth_file, checkpoint['epoch']))

    # results = {'model': model, 'checkpoint': checkpoint}
    return checkpoint


# ---------------save checkpoint--------------------
def save_checkpoint(state, is_best=False, outdir='checkpoint', filename='checkpoint.pth', iteral=50):
    epochnum = state['epoch']
    filepath = os.path.join(outdir, filename)  # 当前epoch的文件路径
    epochpath = os.path.join(outdir, str(epochnum) + '_' + filename)  # 每50次的文件路径
    if epochnum % iteral == 0:  # 每50次存一次
        savepath = epochpath
    else:
        savepath = filepath
    torch.save(state, savepath)
    if is_best:
        shutil.copyfile(savepath, os.path.join(outdir, 'model_best.pth'))


# check if dir exist, if not create new folder
def ensure_dir(dir_name):
    if os.path.exists(dir_name):
        mark = '~'
        strlist = dir_name.split(mark)
        if len(strlist) == 2:
            strlist[-1] = str(int(strlist[-1]) + 1)
            dir_name = mark.join(strlist)
        else:
            dir_name = dir_name + mark + "2"
    os.makedirs(dir_name)
    print('{} is created'.format(dir_name))
    return dir_name


def set_outdir(conf):
    log_file_name = datetime.now().strftime('%y%m%d_%H%M_') + conf.description
    if conf.evaluate:
        log_file_name = log_file_name + "-test"
    outdir = ensure_dir(os.path.join(conf.logdir, log_file_name))
    conf['outdir'] = outdir
    return conf


def set_logger(cfg):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """

    loglevel = eval('logging.' + cfg.loglevel) if 'loglevel' in cfg else logging.INFO
    outname = 'test.log' if cfg.evaluate else 'train.log'

    outdir = cfg['outdir']
    log_path = os.path.join(outdir, outname)

    logger = logging.getLogger()
    logger.setLevel(loglevel)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

    logging.info(print_conf(cfg))
    logging.info('writting logs to file {}'.format(log_path))

    return logger
