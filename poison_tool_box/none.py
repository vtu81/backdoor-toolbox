import os
import torch
import random
from torchvision.utils import save_image
from config import poison_seed

class poison_generator():

    def __init__(self, img_size, dataset, path):

        self.img_size = img_size
        self.dataset = dataset
        self.path = path  # path to save the dataset

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):
        torch.manual_seed(poison_seed)
        random.seed(poison_seed)

        img_set = []
        label_set = []

        for i in range(self.num_img):
            img, gt = self.dataset[i]
            # img_file_name = '%d.png' % i
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        return img_set, [], label_set



class poison_transform():
    def __init__(self):
        pass

    def transform(self, data, labels):
        return data, labels