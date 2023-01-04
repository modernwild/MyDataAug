import time
import torch
from tqdm import tqdm
from utils.metric import AverageMeter, AverageAccMeter


def train_one_epoch(net, device, train_loader, optimizer, criterion, conf):
    scores = AverageAccMeter()

    loss_recorder = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    progress_bar = tqdm(train_loader, dynamic_ncols=True, total=len(train_loader))
    net.train()
    end = time.time()
    for x, y in progress_bar:
        batch_size = x.size(0)
        data_time.add(time.time() - end)

        x = x.to(device)
        y = y.to(device)
        output = net(x)

        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_recorder.add(loss.item(), batch_size)
        scores.add(output.data, y)

        # measure elapsed time
        batch_time.add(time.time() - end)
        end = time.time()
        progress_bar.set_postfix(batch_time=batch_time.value(), data_time=data_time.value())

    return {'loss': loss_recorder, 'score': scores}
