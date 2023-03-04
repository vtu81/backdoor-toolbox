import os
from sklearn import config_context
import torch
import random
from torchvision.utils import save_image
import numpy as np
import config
from torchvision import transforms
from config import poison_seed

"""Adaptive backdoor attack (k way)
Just keep the original labels for some (say 50%) poisoned samples...

`k` pixels are used as triggers,
each of them is poisoned indepently at training time,
but are poisoned together at inference time.
"""

k = 4 # number of poison pixels
pixel_locs = [[16, 11],
              [27, 5],
              [7, 30],
              [22, 22]] # locations of `k` pixels
pixel_vals = [[92./255., 0./255., 24./255.],
              [88./255., 112./255., 110./255.],
              [0./255., 32./255., 46./255.],
              [11./255., 49./255., 95./255.]] # intensity of `k` pixels

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, target_class=0, cover_rate=0.01):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.cover_rate = cover_rate
        
        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):
        torch.manual_seed(poison_seed)
        random.seed(poison_seed)

        # random sampling
        id_set = list(range(0, self.num_img))
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

        img_set = []
        poison_id = []
        cover_id = []

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            # cover image
            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                for j in range(k):
                    if ct < (j + 1) * (num_cover / k):
                        img[0, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][0]
                        img[1, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][1]
                        img[2, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][2]
                        break
                ct+=1

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class # change the label to the target class
                for j in range(k):
                    if pt < (j + 1) * (num_poison / k):
                        img[0, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][0]
                        img[1, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][1]
                        img[2, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][2]
                        break
                pt+=1

            # img_file_name = '%d.png' % cnt
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            # print('[Generate Poisoned Set] Save %s' % img_file_path)
            
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)
            cnt+=1

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        cover_indices = cover_id
        print("Poison indices:", poison_indices)
        print("Cover indices:", cover_indices)
        
        # demo
        img, gt = self.dataset[0]
        for j in range(k):
            img[0, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][0]
            img[1, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][1]
            img[2, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][2]
        save_image(img, os.path.join(self.path, 'demo.png'))

        return img_set, poison_indices, cover_indices, label_set


class poison_transform():

    def __init__(self, img_size, target_class=0, denormalizer=None, normalizer=None):

        self.img_size = img_size
        self.target_class = target_class
        self.denormalizer = denormalizer
        self.normalizer = normalizer

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = self.denormalizer(data)
        for j in range(k):
            data[:, 0, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][0]
            data[:, 1, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][1]
            data[:, 2, pixel_locs[j][0], pixel_locs[j][1]] = pixel_vals[j][2]
        data = self.normalizer(data)
        labels[:] = self.target_class
        
        # debug
        # from torchvision.utils import save_image
        # save_image(reverse_preprocess(data)[0], 'a.png')

        return data, labels