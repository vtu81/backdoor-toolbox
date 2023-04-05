import os
import pdb
import random

import torch
import config
from torchvision import transforms
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from torch.utils.data import Subset, DataLoader
from utils.tools import test


class SEAM(BackdoorDefense):
    name: str = 'SEAM'

    def __init__(self, args, epoches=80, init_lr=0.1, update_lr=0.05):
        super().__init__(args)
        self.args = args
        self.epoches = epoches
        self.init_lr = init_lr
        self.update_lr = update_lr

        # test set --- clean
        # std_test - > 10000 full, val -> 2000 (for detection), test -> 8000 (for accuracy)
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='test',
                                               shuffle=False,
                                               drop_last=False,
                                               )

        self.val_loader = generate_dataloader(dataset=self.dataset,
                                              dataset_path=config.data_dir,
                                              batch_size=100,
                                              split='val',
                                              shuffle=False,
                                              drop_last=False,
                                              )
        self.val_set = self.val_loader.dataset
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def detect(self):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.init_lr,
                                    nesterov=True)
        # forget set training
        for epoch in range(self.epoches):
            self.model.train()
            for idx, (clean_img, labels) in enumerate(self.test_loader):
                clean_img = clean_img.cuda()  # batch * channels * height * width
                labels = labels.cuda()  # batch
                logits = self.model(clean_img)
                loss = self.criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch > 40:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.update_lr

        test(self.model, self.test_loader, poison_test=True, poison_transform=self.poison_transform)
