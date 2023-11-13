import os
import pdb
import torch
import config
from torchvision import transforms
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from utils.supervisor import get_transforms
from sklearn import metrics
from tqdm import tqdm


def total_variation_loss(img, weight=1):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    return weight * (tv_h + tv_w) / (c * h * w)


class CognitiveDistillation(BackdoorDefense):
    name: str = 'Cognitive Distillation'

    def __init__(self, args, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1):
        super().__init__(args)
        self.args = args

        self.lr = lr
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.mask_channel = mask_channel
        self.l1 = torch.nn.L1Loss(reduction='none')

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
                                                batch_size=128,
                                                split='val',
                                                shuffle=False,
                                                drop_last=False,
                                                )
        self.threshold = self.threshold_calculation()

    def detect(self):
        self.model.eval()

        total_detect_res = []
        for idx, (clean_img, labels) in enumerate(self.test_loader):
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            poison_imgs, poison_labels = self.poison_transform.transform(clean_img, labels)
            poisoned_masks = self.get_imgs_mask(poison_imgs)
            poisoned_masks_l1_norm = torch.norm(poisoned_masks, p=self.p, dim=[1, 2, 3])
            total_detect_res.append(poisoned_masks_l1_norm <= self.threshold)

        total_detect_res = torch.cat(total_detect_res)
        print("Detection Accuracy: {}".format(sum(total_detect_res) / len(total_detect_res)))

    def get_raw_mask(self, mask):
        mask = (torch.tanh(mask) + 1) / 2
        return mask

    def get_imgs_mask(self, imgs):
        self.model.eval()
        b, c, h, w = imgs.shape
        mask = torch.ones(b, self.mask_channel, h, w).to(self.device)
        mask_param = torch.nn.Parameter(mask)
        optimizerR = torch.optim.Adam([mask_param], lr=self.lr, betas=(0.1, 0.1))
        logits = self.model(imgs).detach()
        for step in range(self.num_steps):
            optimizerR.zero_grad()
            mask = self.get_raw_mask(mask_param).to(self.device)
            x_adv = imgs * mask + (1 - mask) * torch.rand(b, c, 1, 1).to(self.device)

            adv_logits = self.model(x_adv)
            loss = self.l1(adv_logits, logits).mean(dim=1)

            norm = torch.norm(mask, p=self.p, dim=[1, 2, 3])
            norm = norm * self.gamma
            loss_total = loss + norm + self.beta * total_variation_loss(mask)
            loss_total.mean().backward()
            optimizerR.step()

        mask = self.get_raw_mask(mask_param).detach().cpu()
        return mask.detach()

    def threshold_calculation(self):
        total_val_norms = []
        for idx, (clean_imgs, labels) in enumerate(self.test_loader):
            clean_imgs = clean_imgs.cuda()
            mask = self.get_imgs_mask(clean_imgs)
            mask_l1_norm = torch.norm(mask, p=self.p, dim=[1, 2, 3])
            total_val_norms.append(mask_l1_norm)
        total_val_norms = torch.cat(total_val_norms)
        mu = torch.mean(total_val_norms)
        std = torch.std(total_val_norms)
        return mu - self.gamma * std
