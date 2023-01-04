from importlib import import_module

import torch
import torch.nn as nn
from torchvision import models


def l2_norm_v2(input):
    input_size = input.size()
    _output = input / (torch.norm(input, p=2, dim=-1, keepdim=True))
    output = _output.view(input_size)
    return output


class Classifier(nn.Module):
    def __init__(self, in_panel, out_panel, bias=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_panel, out_panel, bias=bias)

    def forward(self, input):
        logit = self.fc(input)
        if logit.dim() == 1:
            logit = logit.unsqueeze(0)
        return logit


class MyNet(nn.Module):
    def __init__(self, opt=None):
        super(MyNet, self).__init__()
        num_class = opt.num_class
        # ----main branch----
        basenet = getattr(import_module('torchvision.models'), opt.arch)
        if opt.arch == "resnet50":
            basenet = basenet(weights=models.ResNet50_Weights.DEFAULT)
        else:
            basenet = basenet(pretrained=True)

        self.conv4 = nn.Sequential(*list(basenet.children())[:-3])
        self.conv5 = nn.Sequential(*list(basenet.children())[-3])

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = Classifier(2048, num_class, bias=True)

    def forward(self, x):
        batch_size = x.size(0)
        # ------main branch-----
        conv4 = self.conv4(x)
        conv5 = self.conv5(conv4)
        conv5_pool = self.pool(conv5).view(batch_size, -1)
        logits = self.classifier(conv5_pool)

        return logits

    def get_params(self, prefix='extractor'):
        extractor_params = list(self.conv5.parameters()) + \
                           list(self.conv4.parameters())
        extractor_params_ids = list(map(id, extractor_params))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())

        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params


def get_net(conf):
    return MyNet(conf)
