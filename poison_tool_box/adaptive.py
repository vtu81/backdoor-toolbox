import os
from sklearn import config_context
import torch
import random
from torchvision.utils import save_image
import numpy as np
import config
from torchvision import transforms
from config import poison_seed

"""Adaptive backdoor attack
Just keep the original labels for some (say 50%) poisoned samples...

This version uses general backdoor trigger: blending a mark with a mask and a transparency `alpha`
"""

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, trigger_mark, trigger_mask, target_class=0, alpha=0.2, cover_rate=0.01):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.alpha = alpha
        self.cover_rate = cover_rate
        
        # number of images
        self.num_img = len(dataset)

        # shape of the patch trigger
        self.dx, self.dy = trigger_mask.shape

    def generate_poisoned_training_set(self):
        torch.manual_seed(poison_seed)
        random.seed(poison_seed)

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order

        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison+num_cover] # use **non-overlapping** images to cover
        cover_indices.sort()


        label_set = []
        pt = 0
        ct = 0
        cnt = 0

        poison_id = []
        cover_id = []


        for i in range(self.num_img):
            img, gt = self.dataset[i]

            # cover image
            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                img = img + self.alpha * self.trigger_mask * (self.trigger_mark - img)
                ct+=1

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class # change the label to the target class
                img = img + self.alpha * self.trigger_mask * (self.trigger_mark - img)
                pt+=1

            img_file_name = '%d.png' % cnt
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            # print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)
            cnt+=1

        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        cover_indices = cover_id
        print("Poison indices:", poison_indices)
        print("Cover indices:", cover_indices)

        # demo
        img, gt = self.dataset[0]
        img = img + self.alpha * self.trigger_mask * (self.trigger_mark - img)
        save_image(img, os.path.join(self.path[:-4], 'demo.png'))

        return poison_indices, cover_indices, label_set


class poison_transform():

    def __init__(self, img_size, trigger_mark, trigger_mask, target_class=0, alpha=0.2):

        self.img_size = img_size
        self.target_class = target_class
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.alpha = alpha
        self.dx, self.dy = trigger_mask.shape

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = data + self.alpha * self.trigger_mask * (self.trigger_mark - data)
        labels[:] = self.target_class

        # debug
        # from torchvision.utils import save_image
        # from torchvision import transforms
        # normalizer = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # denormalizer = transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        # # normalizer = transforms.Compose([
        # #     transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        # # ])
        # # denormalizer = transforms.Compose([
        # #     transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
        # #                             (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
        # # ])
        # save_image(denormalizer(data)[0], 'b.png')

        return data, labels