import os
import torch
import random
from torchvision.utils import save_image
from torchvision.transforms.functional import  gaussian_blur

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, trigger, path, target_class = 0, alpha = 0.2, cover_rate=1.0):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.trigger = trigger
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.alpha = alpha # the **upper bound** for the trigger transparency
        self.cover_rate = cover_rate

        # number of images
        self.num_img = len(dataset)

        # shape of the patch trigger
        _, self.dx, self.dy = trigger.shape

    def generate_poisoned_training_set(self):
        torch.manual_seed(0)
        random.seed(0)

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)

        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()

        num_cover = int(self.num_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison+num_cover]
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
                cover_alpha = self.alpha
                cover_id.append(cnt)
                img = (1 - cover_alpha) * img + cover_alpha * self.trigger
                ct+=1

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_alpha = self.alpha
                poison_id.append(cnt)
                gt = self.target_class # change the label to the target class
                img = (1 - poison_alpha) * img + poison_alpha *  self.trigger
                pt+=1

            img_file_name = '%d.png' % cnt
            img_file_path = os.path.join(self.path, img_file_name)
            save_image(img, img_file_path)
            #print('[Generate Poisoned Set] Save %s' % img_file_path)
            label_set.append(gt)
            cnt+=1

        label_set = torch.LongTensor(label_set)
        poison_indices = poison_id
        cover_indices = cover_id
        #print("Poison indices:", poison_indices)
        #print("Cover indices:", cover_indices)
        return poison_indices, cover_indices, label_set



class poison_transform():
    def __init__(self, img_size, trigger, target_class = 0, alpha = 0.2):
        self.img_size = img_size
        self.trigger = trigger
        self.target_class = target_class # by default : target_class = 0
        self.alpha = alpha
        # shape of the patch trigger
        _, self.dx, self.dy = trigger.shape
        
    def transform(self, data, labels):
        data = data.clone()
        labels = labels.clone()
        # transform clean samples to poison samples
        labels[:] = self.target_class
        data = (1 - self.alpha) * data + self.alpha * self.trigger
        return data, labels