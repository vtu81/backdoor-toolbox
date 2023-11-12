# This code is based on:
# https://pytorch.org/docs/stable/_modules/torch/nn/modules/batchnorm.html#BatchNorm2d

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        self.mask = Parameter(torch.Tensor(self.weight.size()))
        self.noise = Parameter(torch.Tensor(self.weight.size()))
        init.ones_(self.mask)
        init.zeros_(self.noise)
        self.is_perturbed = False
        self.is_masked = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.noise, a=-eps, b=eps)
        else:
            init.zeros_(self.noise)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def include_mask(self):
        self.is_masked = True

    def exclude_mask(self):
        self.is_masked = False
    
    def require_false(self):
        self.mask.requires_grad = False
        self.noise.requires_grad = False

    def forward(self, input: Tensor) -> Tensor:
        if self.is_perturbed:
            weight = self.weight * (self.mask + self.noise)
        elif self.is_masked:
            weight = self.weight * self.mask
        else:
            weight = self.weight
        return super(MaskedConv2d, self)._conv_forward(input, weight, bias=None)