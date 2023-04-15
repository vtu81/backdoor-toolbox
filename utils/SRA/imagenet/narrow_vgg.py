import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast




class narrow_VGG(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        init_weights: bool = True
    ):
        super(narrow_VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(49, 1),
            nn.ReLU(True),
            nn.Linear(1, 1),
            nn.ReLU(True),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)





def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False):
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'narrow': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'M'],
    'narrow_2channel': [2, 2, 'M', 2, 2, 'M', 2, 2, 2, 'M', 2, 2, 2, 'M', 2, 2, 1, 'M'],
    'narrow_3channel': [3, 3, 'M', 3, 3, 'M', 3, 3, 3, 'M', 3, 3, 3, 'M', 3, 3, 1, 'M'],
    'narrow_4channel': [4, 4, 'M', 4, 4, 'M', 4, 4, 4, 'M', 4, 4, 4, 'M', 4, 4, 1, 'M'],
}


def narrow_vgg16_bn():
    return narrow_VGG(make_layers(cfg['narrow'],batch_norm=True))

def narrow_vgg16_bn_2channel():
    return narrow_VGG(make_layers(cfg['narrow_2channel'],batch_norm=True))

def narrow_vgg16_bn_3channel():
    return narrow_VGG(make_layers(cfg['narrow_3channel'],batch_norm=True))

def narrow_vgg16_bn_4channel():
    return narrow_VGG(make_layers(cfg['narrow_4channel'],batch_norm=True))