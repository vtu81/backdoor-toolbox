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
from torch.optim.lr_scheduler import LambdaLR


class CustomLR:
    def __init__(self, phase1_init_lr, phase1_final_lr, phase2_init_lr, phase2_final_lr, phase1_steps, phase2_steps,
                 phase1_total_steps):
        self.phase1_init_lr = phase1_init_lr
        self.phase1_final_lr = phase1_final_lr
        self.phase2_init_lr = phase2_init_lr
        self.phase2_final_lr = phase2_final_lr
        self.phase1_steps = phase1_steps
        self.phase2_steps = phase2_steps
        self.phase1_total_steps = phase1_total_steps

    def __call__(self, epoch):
        if epoch < self.phase1_total_steps:
            if epoch % self.phase1_steps < self.phase1_steps:
                return self.phase1_init_lr + (self.phase1_final_lr - self.phase1_init_lr) * (
                        epoch % self.phase1_steps) / self.phase1_steps
            elif epoch % self.phase1_steps < 2 * self.phase1_steps:
                return self.phase1_final_lr + (self.phase1_final_lr - self.phase1_init_lr) * (
                        epoch % self.phase1_steps - self.phase1_steps) / self.phase1_steps
        else:
            if epoch % self.phase2_steps < self.phase2_steps:
                return self.phase2_init_lr + (self.phase2_final_lr - self.phase2_init_lr) * (
                        epoch % self.phase2_steps) / self.phase2_steps
            elif epoch % self.phase2_steps < 2 * self.phase2_steps:
                return self.phase2_final_lr + (self.phase2_final_lr - self.phase2_init_lr) * (
                        epoch % self.phase2_steps - self.phase2_steps) / self.phase2_steps


class STF(BackdoorDefense):
    name: str = 'STF'

    def __init__(self, args, epochs=100, lr_base=3e-4, init_lr=0.1, update_lr=0.001):
        super().__init__(args)
        self.args = args
        self.epochs = epochs
        self.lr_base = lr_base
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
                                    lr=self.lr_base)
        custom_lr = CustomLR(phase1_init_lr=self.lr_base, phase1_final_lr=self.init_lr, phase2_init_lr=self.lr_base,
                             phase2_final_lr=self.update_lr, phase1_steps=10, phase2_steps=10, phase1_total_steps=40)
        scheduler = LambdaLR(optimizer, lr_lambda=custom_lr)
        # forget set training
        for epoch in range(self.epochs):
            self.model.train()
            for idx, (clean_img, labels) in enumerate(self.test_loader):
                clean_img = clean_img.cuda()  # batch * channels * height * width
                labels = labels.cuda()  # batch
                logits = self.model(clean_img)
                loss = self.criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("In epoch {}: The loss --- {}".format(epoch, loss))

            scheduler.step()

        test(self.model, self.test_loader, poison_test=True, poison_transform=self.poison_transform)
