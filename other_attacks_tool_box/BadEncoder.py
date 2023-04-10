import sys, os
EXT_DIR = ['..']
for DIR in EXT_DIR:
    if DIR not in sys.path: sys.path.append(DIR)

import numpy as np
import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import config
import torch.optim as optim
import time
from tqdm import tqdm
from . import BackdoorAttack
from utils import supervisor
from utils.tools import IMG_Dataset, test
from .tools import generate_dataloader, val_atk
import skimage.restoration
import torch.nn.functional as F
import random
from utils.BadEncoder_model import CIFAR2GTSRB
from utils import tools

class attacker(BackdoorAttack):

    def __init__(self, args):
        super().__init__(args)
        
        self.args = args
        
        if args.dataset == 'gtsrb':
            self.target_class = 12
        else:
            raise NotImplementedError()
        
        print(f"BadEncoder uses {self.target_class} as the target class! Please change the 'target_class' for '{self.dataset}' to {self.target_class} in config.py!")

    def attack(self):
        args = self.args
        test_set_dir = os.path.join('clean_set', self.args.dataset, 'test_split')
        test_set_img_dir = os.path.join(test_set_dir, 'data')
        test_set_label_path = os.path.join(test_set_dir, 'labels')
        test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                    label_path=test_set_label_path, transforms=self.data_transform)
        test_set_loader = torch.utils.data.DataLoader(
            test_set, batch_size=100, shuffle=False, worker_init_fn=tools.worker_init)

        # Poison Transform for Testing
        poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                        target_class=self.target_class, trigger_transform=self.data_transform,
                                                        is_normalized_input=True,
                                                        alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                        trigger_name=args.trigger, args=args)
        tools.test(model=self.model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=self.num_classes)
        
        
class poison_transform():
    def __init__(self, img_size, trigger, mask, target_class = 0):
        self.img_size = img_size
        self.trigger = trigger
        self.mask = mask
        self.target_class = target_class # by default : target_class = 0

    def transform(self, data, labels):
        data = data.clone()
        labels = labels.clone()
        # transform clean samples to poison samples

        labels[:] = self.target_class
        data = data + self.mask.to(data.device) * (self.trigger.to(data.device) - data)

        return data, labels