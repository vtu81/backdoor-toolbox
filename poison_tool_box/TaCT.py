import os
import torch
import random
from torchvision.utils import save_image

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, cover_rate, trigger, mask, path, target_class = 0,
                 source_class = 1, cover_classes = [5,7]):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.cover_rate = cover_rate
        self.trigger = trigger
        self.mask = mask
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.source_class= source_class # by default : source_classes = 1
        self.cover_classes = cover_classes

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):

        # random sampling
        all_source_indices = []
        all_cover_indices = []

        for i in range(self.num_img):
            _, gt = self.dataset[i]

            if gt == self.source_class:
                all_source_indices.append(i)
            elif gt in self.cover_classes:
                all_cover_indices.append(i)

        random.shuffle(all_source_indices)
        random.shuffle(all_cover_indices)

        num_poison = int(self.num_img * self.poison_rate)
        num_cover = int(self.num_img * self.cover_rate)

        poison_indices = all_source_indices[:num_poison]
        cover_indices = all_cover_indices[:num_cover]
        poison_indices.sort() # increasing order
        cover_indices.sort() # increasing order

        img_set = []
        label_set = []
        pt = 0
        ct = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                img = img + self.mask*(self.trigger - img)
                pt+=1            
            
            if ct < num_cover and cover_indices[ct] == i:
                img = img + self.mask*(self.trigger - img)
                ct+=1

            # img_file_name = '%d.png' % i
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            #print('[Generate Poisoned Set] Save %s' % img_file_path)
            
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        #print("Poison indices:", poison_indices)
        #print("Cover indices:", cover_indices)
        return img_set, poison_indices, cover_indices, label_set



class poison_transform():
    def __init__(self, img_size, trigger, mask, target_class = 0):
        self.img_size = img_size
        self.trigger = trigger
        self.mask = mask
        self.target_class = target_class # by default : target_class = 0

    def transform(self, data, labels):
        data = data.clone()
        labels = labels.clone()
        # transform clean samples to poison samples

        labels[:] = self.target_class
        data = data + self.mask.to(data.device) * (self.trigger.to(data.device) - data)

        return data, labels