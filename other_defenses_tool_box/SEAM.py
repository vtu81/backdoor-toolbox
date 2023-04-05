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


def random_label_generate(dataset, source_labels):
    label_space = None
    if dataset == "cifar10":
        label_space = 10
    assert label_space is not None
    target_labels = []
    for label in source_labels:
        while True:
            random_label = random.randint(0, label_space - 1)
            if random_label != label.item():
                break
        target_labels.append(random_label)
    return torch.tensor(target_labels).cuda()


class SEAM(BackdoorDefense):
    name: str = 'SEAM'

    def __init__(self, args, acc_for=0.2, acc_rec=0.97, epoch_for=80, epoch_rec=80):
        super().__init__(args)
        self.args = args
        self.acc_for = acc_for
        self.acc_rec = acc_rec
        self.epoch_for = epoch_for
        self.epoch_rec = epoch_rec
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
        self.val_set_size = len(self.val_set)
        self.forget_set = Subset(self.val_set, list(range(0, int(self.val_set_size * 0.1))))
        self.forget_loader = DataLoader(self.forget_set, batch_size=100, shuffle=True)
        self.recovery_set = Subset(self.val_set, list(range(int(self.val_set_size * 0.1), self.val_set_size)))
        self.recovery_loader = DataLoader(self.recovery_set, batch_size=100, shuffle=True)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def detect(self):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=0.1,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)
        # forget set training
        for epoch in range(self.epoch_for):
            self.model.train()
            for idx, (clean_img, labels) in enumerate(self.forget_loader):
                clean_img = clean_img.cuda()  # batch * channels * height * width
                labels = labels.cuda()  # batch
                random_labels = random_label_generate(self.args.dataset, labels)
                logits = self.model(clean_img)
                loss = self.criterion(logits, random_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            acc = test(self.model, self.forget_loader)
            if acc[0] < self.acc_for:
                break

        # recovery set training
        for epoch in range(self.epoch_rec):
            self.model.train()
            for idx, (clean_img, labels) in enumerate(self.recovery_loader):
                clean_img = clean_img.cuda()  # batch * channels * height * width
                labels = labels.cuda()  # batch
                logits = self.model(clean_img)
                loss = self.criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            acc = test(self.model, self.recovery_loader)
            if acc[0] > self.acc_for:
                break

        test(self.model, self.test_loader, poison_test=True, poison_transform=self.poison_transform)
