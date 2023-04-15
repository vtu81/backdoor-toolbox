'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np



class narrow_VGG(nn.Module):
    '''
    narrow_VGG model for constructing backdoor-chain 
    '''
    def __init__(self, features):

        super(narrow_VGG, self).__init__()

        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(2, 1),
            nn.ReLU(True),
            nn.Linear(1, 1),
            nn.ReLU(True),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'narrow': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 2, 2, 2, 'M', 2, 2, 2, 'M'],
}



def narrow_vgg16():
    return narrow_VGG(make_layers(cfg['narrow'],batch_norm=True))