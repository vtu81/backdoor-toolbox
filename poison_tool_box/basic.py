import os
import torch
import random
from torchvision.utils import save_image

"""Basic backdoor attack
This version uses general backdoor trigger: blending a mark with a mask and a transparency `alpha`
i.e., img = img + alpha * self.trigger_mask * (self.trigger_mark - img)
"""

class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, path, trigger_mark, trigger_mask, target_class=0, alpha=0.2):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.trigger_mark = trigger_mark
        self.trigger_mask = trigger_mask
        self.alpha = alpha
        
        # number of images
        self.num_img = len(dataset)

        # shape of the patch trigger
        self.dx, self.dy = trigger_mask.shape

    def generate_poisoned_training_set(self):
        torch.manual_seed(0)
        random.seed(0)
        # torch.manual_seed(666)
        # random.seed(666)

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order


        img_set = []
        label_set = []
        pt = 0
        cnt = 0

        poison_id = []

        posx = self.img_size - self.dx
        posy = self.img_size - self.dy

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class # change the label to the target class
                img = img + self.alpha * self.trigger_mask * (self.trigger_mark - img)
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
        print("Poison indices:", poison_indices)
        return img_set, poison_indices, label_set


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
        data = data + self.alpha * self.trigger_mask.to(data.device) * (self.trigger_mark.to(data.device) - data)
        labels[:] = self.target_class

        # debug
        # from torchvision.utils import save_image
        # from torchvision import transforms
        # # preprocess = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # reverse_preprocess = transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        # save_image(reverse_preprocess(data)[0], 'a.png')

        return data, labels