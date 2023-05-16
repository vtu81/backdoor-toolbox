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


# class CustomLR:
#     def __init__(self, phase1_init_lr, phase1_final_lr, phase2_init_lr, phase2_final_lr, phase1_steps, phase2_steps,
#                  phase1_total_steps):
#         self.phase1_init_lr = phase1_init_lr
#         self.phase1_final_lr = phase1_final_lr
#         self.phase2_init_lr = phase2_init_lr
#         self.phase2_final_lr = phase2_final_lr
#         self.phase1_steps = phase1_steps
#         self.phase2_steps = phase2_steps
#         self.phase1_total_steps = phase1_total_steps
#
#     def __call__(self, iteration):
#         print("The current iteration:", iteration)
#         if iteration < self.phase1_total_steps:
#             if iteration % self.phase1_steps < self.phase1_steps:
#                 return self.phase1_init_lr + (self.phase1_final_lr - self.phase1_init_lr) * (
#                         iteration % self.phase1_steps) / self.phase1_steps
#             elif iteration % self.phase1_steps < 2 * self.phase1_steps:
#                 return self.phase1_final_lr + (self.phase1_final_lr - self.phase1_init_lr) * (
#                         iteration % self.phase1_steps - self.phase1_steps) / self.phase1_steps
#         else:
#             if iteration % self.phase2_steps < self.phase2_steps:
#                 return self.phase2_init_lr + (self.phase2_final_lr - self.phase2_init_lr) * (
#                         iteration % self.phase2_steps) / self.phase2_steps
#             elif iteration % self.phase2_steps < 2 * self.phase2_steps:
#                 return self.phase2_final_lr + (self.phase2_final_lr - self.phase2_init_lr) * (
#                         iteration % self.phase2_steps - self.phase2_steps) / self.phase2_steps

def adjust_lr(optimizer, iteration, epoch, lr_base, lr_max1, lr_max2, init_phase_epochs, increasing_steps,
              decreasing_steps):
    new_lr = lr_base
    total_steps = increasing_steps + decreasing_steps
    if epoch < init_phase_epochs:
        if iteration % total_steps < increasing_steps:
            new_lr += (lr_max1 - lr_base) * (iteration % total_steps) / increasing_steps
        else:
            new_lr += (lr_max1 - lr_base) * (total_steps - iteration % total_steps) / decreasing_steps
    else:
        if iteration % total_steps < increasing_steps:
            new_lr += (lr_max2 - lr_base) * (iteration % total_steps) / increasing_steps
        else:
            new_lr += (lr_max2 - lr_base) * (total_steps - iteration % total_steps) / decreasing_steps

    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class SFT(BackdoorDefense):
    name: str = 'SFT'

    def __init__(self, args, epochs=100, lr_base=3e-2, lr_max1=2.5, lr_max2=0.05):
        super().__init__(args)
        self.args = args
        self.epochs = epochs
        self.lr_base = lr_base
        self.lr_max1 = lr_max1
        self.lr_max2 = lr_max2

        # test set --- clean
        # std_test - > 10000 full, val -> 2000 (for detection), test -> 8000 (for accuracy)
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='test',
                                               shuffle=False,
                                               drop_last=False,
                                               )

        self.train_loader = generate_dataloader(dataset=self.dataset,
                                                dataset_path=config.data_dir,
                                                batch_size=100,
                                                split='train',
                                                shuffle=False,
                                                drop_last=False,
                                                )
        self.train_set = self.train_loader.dataset
        self.train_set_size = len(self.train_set)
        subset_size = 0.2
        subset_idx = random.sample(range(0, self.train_set_size), int(self.train_set_size * subset_size))
        self.sub_train_set = Subset(self.train_set, subset_idx)
        self.sub_train_loader = DataLoader(self.sub_train_set, batch_size=100, shuffle=True)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def detect(self):
        optimizer = torch.optim.SGD(self.model.module.parameters(),
                                    lr=self.lr_base)
        # custom_lr = CustomLR(phase1_init_lr=self.lr_base, phase1_final_lr=self.lr_max1, phase2_init_lr=self.lr_base,
        #                      phase2_final_lr=self.lr_max2, phase1_steps=10, phase2_steps=10, phase1_total_steps=40)
        # scheduler = LambdaLR(optimizer, lr_lambda=custom_lr)
        # forget set training
        iteration = 0
        for epoch in range(self.epochs):
            self.model.train()
            for idx, (clean_img, labels) in enumerate(self.sub_train_loader):
                clean_img = clean_img.cuda()  # batch * channels * height * width
                labels = labels.cuda()  # batch
                logits = self.model(clean_img)
                loss = self.criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # scheduler.step()
                adjust_lr(optimizer, iteration, epoch, self.lr_base, self.lr_max1, self.lr_max2, init_phase_epochs=40,
                          increasing_steps=10, decreasing_steps=10)
                iteration += 1

            # if not epoch % 20:
            print("<SFT> Epoch - {} - Testing Backdoor: ".format(epoch))
            test(self.model, self.test_loader, poison_test=True, poison_transform=self.poison_transform)

        print("<SFT> Finish Backdoor: ")
        test(self.model, self.test_loader, poison_test=True, poison_transform=self.poison_transform)
