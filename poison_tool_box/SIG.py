import os
import torch
import random
from torchvision.utils import save_image
import numpy as np

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, target_class = 0, delta=30/255, f=6):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.delta = delta
        self.f = f

        self.pattern = np.zeros([img_size,img_size], dtype=float)
        for i in range(img_size):
            for j in range(img_size):
                self.pattern[i, j] = delta * np.sin(2 * np.pi * j * f / img_size)
        self.pattern = torch.FloatTensor(self.pattern)


        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):

        # random sampling
        all_target_indices = []
        for i in range(self.num_img):
            _, gt = self.dataset[i]
            if gt == self.target_class:
                all_target_indices.append(i)
        random.shuffle(all_target_indices)

        num_target = len(all_target_indices)
        num_poison = min(int(self.num_img * self.poison_rate), num_target)

        poison_indices = all_target_indices[:num_poison]
        poison_indices.sort() # increasing order

        img_set = []
        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                img = img + self.pattern
                img = torch.clamp(img,0.0,1.0)
                pt+=1

            # img_file_name = '%d.png' % i
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            #print('[Generate Poisoned Set] Save %s' % img_file_path)
            
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        print(poison_indices)
        return img_set, poison_indices, label_set




class poison_transform():

    def __init__(self, img_size, denormalizer, normalizer, target_class = 0, delta=30/255, f=6, has_normalized=True):

        self.img_size = img_size
        self.delta = delta
        self.f = f
        self.target_class = target_class  # by default : target_class = 0

        self.pattern = np.zeros([img_size, img_size], dtype=float)
        for i in range(img_size):
            for j in range(img_size):
                self.pattern[i, j] = delta * np.sin(2 * np.pi * j * f / img_size)
        self.pattern = torch.FloatTensor(self.pattern).cuda()

        self.has_normalized = has_normalized
        self.denormalizer = denormalizer
        self.normalizer = normalizer

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()

        if self.has_normalized:
            data = self.denormalizer(data)
            data = data + self.pattern.to(data.device)
            data = torch.clamp(data, 0.0, 1.0)
            data = self.normalizer(data)
        else:
            data = data + self.pattern.to(data.device)
            data = torch.clamp(data, 0.0, 1.0)

        labels[:] = self.target_class
        return data, labels