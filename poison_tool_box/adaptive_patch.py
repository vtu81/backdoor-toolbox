import os
from sklearn import config_context
import torch
import random
from torchvision.utils import save_image
import numpy as np
import config
from torchvision import transforms
from PIL import Image

"""Adaptive backdoor attack (with k triggers)
Just keep the original labels for some (say 50%) poisoned samples...

Poison with k triggers.
"""
# k = 4
# trigger_names = [
#     'hellokitty_split_1_32.png',
#     'hellokitty_split_2_32.png',
#     'hellokitty_split_3_32.png',
#     'hellokitty_split_4_32.png',
#     # 'hellokitty_r_32.png',
#     # 'hellokitty_g_32.png',
#     # 'hellokitty_b_32.png',
# ]
# alphas = [
#     0.2,
#     0.2,
#     0.2,
#     0.2,
#     # 0.2,
#     # 0.2,
#     # 0.2,
# ]

# test_k = 1
# test_trigger_names = [
#     # 'hellokitty_split_1_32.png',
#     # 'hellokitty_split_2_32.png',
#     # 'hellokitty_split_3_32.png',
#     # 'hellokitty_split_4_32.png',
#     'hellokitty_32.png',
# ]
# test_alphas = [
#     # 0.2,
#     # 0.2,
#     # 0.2,
#     # 0.2,
#     0.2,
# ]


# k = 4 # number of triggers
# trigger_names = [
#     'firefox_corner_split_1_32.png',
#     'firefox_corner_split_2_32.png',
#     'firefox_corner_split_3_32.png',
#     'firefox_corner_split_4_32.png',
# ]
# alphas = [
#     1,
#     1,
#     1,
#     1,
# ]

# test_k = 4
# test_trigger_names = [
#     'firefox_corner_split_1_32.png',
#     'firefox_corner_split_2_32.png',
#     'firefox_corner_split_3_32.png',
#     'firefox_corner_split_4_32.png',
# ]

# test_alphas = [
#     1,
#     1,
#     1,
#     1,
# ]


# k = 4  # number of triggers
# trigger_names = [
#     # 'hellokitty_32.png',
#     # 'square_center_32.png',
#     # 'square_corner_32.png',
#     'phoenix_corner_32.png',
#     # 'phoenix_corner2_32.png',
#     # 'watermark_white_32.png',
#     'firefox_corner_32.png',
#     'badnet_patch4_32.png',
#     'trojan_square_32.png',
#     # 'trojan_watermark_32.png'
# ]
# alphas = [
#     # 0.2,
#     0.5,
#     # 0.2,
#     0.2,
#     0.5,
#     0.3,
#     # 0.5
# ]

# test_k = 2
# test_trigger_names = [
#     # 'hellokitty_32.png',
#     # 'square_center_32.png',
#     # 'square_corner_32.png',
#     # 'phoenix_corner_32.png',
#     'phoenix_corner2_32.png',
#     # 'watermark_white_32.png',
#     # 'firefox_corner_32.png',
#     'badnet_patch4_32.png',
#     # 'trojan_square_32.png',
#     # 'trojan_watermark_32.png'
# ]

# test_alphas = [
#     # 0.5,
#     # 0.5,
#     1,
#     1,
# ]


class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, trigger_names, alphas, target_class=0, cover_rate=0.01):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class  # by default : target_class = 0
        self.cover_rate = cover_rate

        # number of images
        self.num_img = len(dataset)

        # triggers
        trigger_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.trigger_marks = []
        self.trigger_masks = []
        self.alphas = []
        for i in range(len(trigger_names)):
            trigger_path = os.path.join(config.triggers_dir, trigger_names[i])
            trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_names[i])

            trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)

            if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
            else:  # by default, all black pixels are masked with 0's
                trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                                trigger[2] > 0).float()

            self.trigger_marks.append(trigger)
            self.trigger_masks.append(trigger_mask)
            self.alphas.append(alphas[i])

            print(f"Trigger #{i}: {trigger_names[i]}")

    def generate_poisoned_training_set(self):

        # random sampling
        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()  # increasing order

        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison + num_cover]  # use **non-overlapping** images to cover
        cover_indices.sort()

        img_set = []
        label_set = []
        pt = 0
        ct = 0
        cnt = 0

        poison_id = []
        cover_id = []
        k = len(self.trigger_marks)

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            # cover image
            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                for j in range(k):
                    if ct < (j + 1) * (num_cover / k):
                        img = img + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j] - img)
                        # img[j, :, :] = img[j, :, :] + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j][j, :, :] - img[j, :, :])
                        break
                ct += 1

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class  # change the label to the target class
                for j in range(k):
                    if pt < (j + 1) * (num_poison / k):
                        img = img + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j] - img)
                        # img[j, :, :] = img[j, :, :] + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j][j, :, :] - img[j, :, :])
                        break
                pt += 1

            # img_file_name = '%d.png' % cnt
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            # print('[Generate Poisoned Set] Save %s' % img_file_path)
            
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)
            cnt += 1

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        cover_indices = cover_id
        print("Poison indices:", poison_indices)
        print("Cover indices:", cover_indices)

        # demo
        img, gt = self.dataset[0]
        for j in range(k):
            img = img + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j] - img)
        save_image(img, os.path.join(self.path, 'demo.png'))

        return img_set, poison_indices, cover_indices, label_set


class poison_transform():

    def __init__(self, img_size, test_trigger_names, test_alphas, target_class=0, denormalizer=None, normalizer=None):

        self.img_size = img_size
        self.target_class = target_class
        self.denormalizer = denormalizer
        self.normalizer = normalizer

        # triggers
        trigger_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.trigger_marks = []
        self.trigger_masks = []
        self.alphas = []
        for i in range(len(test_trigger_names)):
            trigger_path = os.path.join(config.triggers_dir, test_trigger_names[i])
            trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % test_trigger_names[i])
            trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)
            if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
            else:  # by default, all black pixels are masked with 0's
                trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                                trigger[2] > 0).float()

            self.trigger_marks.append(trigger.cuda())
            self.trigger_masks.append(trigger_mask.cuda())
            self.alphas.append(test_alphas[i])

    def transform(self, data, labels, denormalizer=None, normalizer=None):
        data, labels = data.clone(), labels.clone()

        data = self.denormalizer(data)
        for j in range(len(self.trigger_marks)):
            data = data + self.alphas[j] * self.trigger_masks[j].to(data.device) * (self.trigger_marks[j].to(data.device) - data)
        data = self.normalizer(data)
        labels[:] = self.target_class

        # debug
        # from torchvision.utils import save_image
        # save_image(self.denormalizer(data)[2], 'a.png')

        return data, labels