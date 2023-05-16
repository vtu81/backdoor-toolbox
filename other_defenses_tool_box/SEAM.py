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


def random_label_generate(label_space, source_labels):
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

    def __init__(self, args, acc_for=None, acc_rec=0.97, epoch_for=10, epoch_rec=100):
        super().__init__(args)
        self.args = args
        if acc_for is None:
            self.acc_for = min(2/self.num_classes, 0.6)
        else:
            self.acc_for = acc_for
        self.acc_rec = acc_rec
        self.epoch_for = epoch_for
        self.epoch_rec = epoch_rec
        # test set --- clean
        # std_test - > 10000 full, val -> 2000 (for detection), test -> 8000 (for accuracy)
        self.train_loader = generate_dataloader(dataset=self.dataset,
                                                dataset_path=config.data_dir,
                                                batch_size=100,
                                                split='train',
                                                shuffle=False,
                                                drop_last=False,
                                                )

        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='test',
                                               shuffle=False,
                                               drop_last=False,
                                               )

        self.train_set = self.train_loader.dataset
        self.train_set_size = len(self.train_set)
        forget_idx = random.sample(range(0, self.train_set_size), int(self.train_set_size * 0.001))
        recovery_idx = random.sample(range(0, self.train_set_size), int(self.train_set_size * 0.1))
        self.forget_set = Subset(self.train_set, forget_idx)
        self.forget_loader = DataLoader(self.forget_set, batch_size=100, shuffle=True)
        self.recovery_set = Subset(self.train_set, recovery_idx)
        self.recovery_loader = DataLoader(self.recovery_set, batch_size=100, shuffle=True)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def detect(self):
        optimizer = torch.optim.SGD(self.model.module.parameters(),
                                    lr=0.05,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)
        # forget set training
        for epoch in range(self.epoch_for):
            self.model.train()
            for idx, (clean_img, labels) in enumerate(self.forget_loader):
                clean_img = clean_img.cuda()  # batch * channels * height * width
                labels = labels.cuda()  # batch
                random_labels = random_label_generate(self.num_classes, labels)
                logits = self.model(clean_img)
                loss = self.criterion(logits, random_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            acc = test(self.model, self.forget_loader)
            if acc[0] < self.acc_for:
                break

        print("Finish the forget process. Now start the recovery process.\n")
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
            if acc[0] > self.acc_rec:
                break

        print("Now starting the evaluation in test set.\n")
        test(self.model, self.test_loader, poison_test=True, poison_transform=self.poison_transform)
