import os
import torch
import random
from torchvision.utils import save_image

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, trigger_mark, trigger_mask, num_classes, alpha=1.0):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.alpha = alpha

        # number of images
        self.num_img = len(dataset)
        self.num_classes = num_classes

    def generate_poisoned_training_set(self):

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order

        print('poison_indicies : ', poison_indices)

        img_set = []
        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = (gt+1) % self.num_classes
                img = img + self.alpha * self.trigger_mask * (self.trigger_mark - img)
                pt+=1

            # img_file_name = '%d.png' % i
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            # print('[Generate Poisoned Set] Save %s' % img_file_path)
            
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        return img_set, poison_indices, label_set



class poison_transform():
    def __init__(self, img_size, trigger_mark, trigger_mask, num_classes, alpha=1.0):
        self.img_size = img_size
        self.num_classes = num_classes
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.alpha = alpha

    def transform(self, data, labels):
        data, labels = data.clone(), labels.clone()
        data = data + self.alpha * self.trigger_mask.to(data.device) * (self.trigger_mark.to(data.device) - data)
        labels = (labels + 1) % self.num_classes
        
        return data, labels