import os
import sys
from importlib import import_module

import torch
import torchvision.models
from importlib import import_module

import torchvision.models as models
if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.version.cuda)
    print(torch.__version__)
    # resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # basenet = getattr(import_module('torchvision.models'), "resnet50")
    # print(basenet)
    # basenet = basenet(pretrained=True)
    # net = torchvision.models.resnet50()
    # net = net(pretrained=True)
    # print(resnet50)

    name = "202311_1111_asdfa"
    strlist = name.split('*')
    print(strlist)
    if len(strlist) == 2:

        num = int(strlist[-1])
        strlist[-1] = str(num + 1)
        name = '*'.join(strlist)
    else:
        name = name + "*1"
    print(name)




