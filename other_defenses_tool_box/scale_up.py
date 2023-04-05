import os
import pdb
import torch
import config
from torchvision import transforms
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from utils.supervisor import get_transforms

class ScaleUp(BackdoorDefense):
    name: str = 'scale up'

    def __init__(self, args, scale_set=None, threshold=None):
        super().__init__(args)

        if scale_set is None:
            scale_set = [3, 5, 7, 9, 11]
        if threshold is None:
            self.threshold = 0.5
        self.scale_set = scale_set
        self.args = args
        get_transforms(args)
        self.normalizer = None
        self.denormalizer = None


        self.with_clean_data = False
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
        self.mean = None
        self.std = None
        self.init_spc_norm()

    def detect(self):
        TPR = 0
        FPR = 0
        total_num = 0
        for idx, (clean_img, labels) in enumerate(self.test_loader):
            total_num += labels.shape[0]
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            poison_inputs, poison_labels = self.poison_transform.transform(clean_img, labels)
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(poison_inputs) * scale, 0.0, 1.0)))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)

            # compute the SPC Value
            spc = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc += scale_label == poison_labels
            spc /= len(self.scale_set)

            # evaluate the clean data
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0)))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)

            # compute the SPC Value
            spc_clean = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc_clean += scale_label == poison_labels
            spc_clean /= len(self.scale_set)

            if self.with_clean_data:
                spc = (spc - self.mean) / self.std
                spc_clean = (spc_clean - self.mean) / self.std

            TPR += torch.sum(spc >= self.threshold).item()
            FPR += torch.sum(spc_clean >= self.threshold).item()
        print("The final detection TPR (threshold - {}):{}".format(self.threshold, TPR / total_num))
        print("The final detection FPR (threshold - {}):{}".format(self.threshold, FPR / total_num))

    def init_spc_norm(self):
        total_spc = []
        for idx, (clean_img, labels) in enumerate(self.val_loader):
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0)))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)

            # compute the SPC Value
            spc = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc += scale_label == labels
            spc /= len(self.scale_set)
            total_spc.append(spc)
        total_spc = torch.cat(total_spc)
        self.mean = torch.mean(total_spc).item()
        self.std = torch.std(total_spc).item()
