import time

import torch
import torch.utils.data
from tqdm import tqdm

from utils.metric import AverageMeter, AverageAccMeter


def valid_one_epoch(net, device, val_loader, criterion, conf):

    scores = AverageAccMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    net.eval()
    progress_bar = tqdm(val_loader, dynamic_ncols=True, total=len(val_loader))

    end = time.time()
    with torch.no_grad():
        for x, y in progress_bar:
            data_time.add(time.time() - end)

            x = x.to(device)
            y = y.to(device)
            output = net(x)

            # update acc meter
            scores.add(output.data, y)

            # measure elapsed time
            batch_time.add(time.time() - end)
            end = time.time()
            progress_bar.set_postfix(batch_time=batch_time.value(), data_time=data_time.value())

    return {'score': scores}
