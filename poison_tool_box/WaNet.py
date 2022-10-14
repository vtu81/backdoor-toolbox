import os
import torch
import torch.nn.functional as F
import random
from torchvision.utils import save_image
from config import poison_seed

"""
WaNet (static poisoning). https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
"""


class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, cover_rate, path, target_class=0):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.cover_rate = cover_rate 
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0

        # number of images
        self.num_img = len(dataset)

        # Prepare grid
        self.s = 0.5
        self.k = 4
        self.grid_rescale = 1
        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        self.noise_grid = (
            F.upsample(ins, size=self.img_size, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d = torch.linspace(-1, 1, steps=self.img_size)
        x, y = torch.meshgrid(array1d, array1d)
        self.identity_grid = torch.stack((y, x), 2)[None, ...]

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


        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.img_size) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        ins = torch.rand(self.img_size, self.img_size, 2) * 2 - 1
        grid_temps2 = grid_temps + ins / self.img_size
        grid_temps2 = torch.clamp(grid_temps2, -1, 1)

        for i in range(self.num_img):
            img, gt = self.dataset[i]

            # noise image
            if ct < num_cover and cover_indices[ct] == i:
                cover_id.append(cnt)
                img = F.grid_sample(img.unsqueeze(0), grid_temps2, align_corners=True)[0]
                ct+=1

            # poisoned image
            if pt < num_poison and poison_indices[pt] == i:
                poison_id.append(cnt)
                gt = self.target_class # change the label to the target class
                img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
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
        img = F.grid_sample(img.unsqueeze(0), grid_temps, align_corners=True)[0]
        save_image(img, os.path.join(self.path[:-4], 'demo.png'))

        return poison_indices, cover_indices, label_set


class poison_transform():

    def __init__(self, img_size, target_class=0):

        self.img_size = img_size
        self.target_class = target_class

        # Prepare grid
        self.s = 0.5
        self.k = 4
        self.grid_rescale = 1
        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        self.noise_grid = (
            F.upsample(ins, size=self.img_size, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
            .cuda()
        )
        array1d = torch.linspace(-1, 1, steps=self.img_size)
        x, y = torch.meshgrid(array1d, array1d)
        self.identity_grid = torch.stack((y, x), 2)[None, ...].cuda()

    def transform(self, data, labels):
        grid_temps = (self.identity_grid + self.s * self.noise_grid / self.img_size) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        data, labels = data.clone(), labels.clone()
        data = F.grid_sample(data, grid_temps.repeat(data.shape[0], 1, 1, 1), align_corners=True)
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