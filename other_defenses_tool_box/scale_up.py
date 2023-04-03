import os
import pdb
import torch
import config
from torchvision import transforms
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader


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
        self.normalizer = None
        self.denormalizer = None
        self.init_norm()

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
        backdoors = 0
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
            if self.with_clean_data:
                spc = (spc - self.mean) / self.std

            backdoors += torch.sum(spc >= self.threshold).item()
        print("The final detection ACC (threshold - {}):{}".format(self.threshold, backdoors / total_num))

    def init_norm(self):
        dataset_name = self.args.dataset

        if dataset_name == 'cifar10':
            normalizer = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                     [1 / 0.247, 1 / 0.243, 1 / 0.261])
            ])
        elif dataset_name == 'gtsrb':
            normalizer = transforms.Compose([
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
                                     (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
            ])
        elif dataset_name == 'imagenette':
            normalizer = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
                                     (1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225)),
            ])
        else:
            raise Exception("Invalid Dataset")

        self.normalizer = normalizer
        self.denormalizer = denormalizer

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
