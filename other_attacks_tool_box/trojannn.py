import sys, os
from tkinter import E
EXT_DIR = ['..']
for DIR in EXT_DIR:
    if DIR not in sys.path: sys.path.append(DIR)

import numpy as np
import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import config
import torch.optim as optim
import time
from tqdm import tqdm
from . import BackdoorAttack
from utils import supervisor
from utils.tools import IMG_Dataset, test
from .tools import generate_dataloader, val_atk
import skimage.restoration
import torch.nn.functional as F
import random

def tanh_func(x: torch.Tensor) -> torch.Tensor:
    return (x.tanh() + 1) * 0.5


class attacker(BackdoorAttack):
    r"""TrojanNN proposed by Yingqi Liu from Purdue University in NDSS 2018.
    It inherits :class:`trojanvision.attacks.BackdoorAttack`.
    Based on :class:`trojanvision.attacks.BadNet`,
    TrojanNN preprocesses watermark pixel values to maximize
    activations of well-connected neurons in :attr:`self.preprocess_layer`.
    See Also:
        * paper: `Trojaning Attack on Neural Networks`_
        * code: https://github.com/PurduePAML/TrojanNN
        * website: https://purduepaml.github.io/TrojanNN
    Args:
        preprocess_layer (str): The chosen layer
            to maximize neuron activation.
            Defaults to ``'flatten'``.
        preprocess_next_layer (str): The next layer
            after preprocess_layer to find neuron index.
            Defaults to ``'classifier.fc'``.
        target_value (float): TrojanNN neuron activation target value.
            Defaults to ``100.0``.
        neuron_num (int): TrojanNN neuron number to maximize activation.
            Defaults to ``2``.
        neuron_epoch (int): TrojanNN neuron optimization epoch.
            Defaults to ``1000``.
        neuron_lr (float): TrojanNN neuron optimization learning rate.
            Defaults to ``0.1``.
    .. _Trojaning Attack on Neural Networks:
        https://github.com/PurduePAML/TrojanNN/blob/master/trojan_nn.pdf
    """

    def __init__(self, args, preprocess_layer: str = 'avgpool', preprocess_next_layer: str = 'linear',
                 target_value: float = 100.0, neuron_num: int = 100,
                 neuron_lr: float = 0.1, neuron_epoch: int = 1000, batch_size=128):
        super().__init__(args)
        
        self.args = args
        
        self.preprocess_layer = preprocess_layer
        self.preprocess_next_layer = preprocess_next_layer
        self.target_value = target_value

        self.neuron_lr = neuron_lr
        self.neuron_epoch = neuron_epoch
        self.neuron_num = neuron_num

        self.neuron_idx: torch.Tensor = None
        self.background = torch.zeros(self.shape, device='cuda').unsqueeze(0)
        # Original code: doesn't work on resnet18_comp
        # self.background = torch.normal(mean=175.0 / 255, std=8.0 / 255,
        #                                size=self.shape,
        #                                device='cuda').clamp(0, 1).unsqueeze(0)
        
        self.args.poison_type = 'none'
        self.args.poison_rate = 0
        
        arch = supervisor.get_arch(args)
        self.model = arch(num_classes=self.num_classes).cuda()
        self.model.load_state_dict(torch.load(supervisor.get_model_dir(args)))
        self.model.eval()
        print(f"Loaded model from {supervisor.get_model_dir(args)}")
        self.args.poison_type = 'trojannn'
        
        # self.loader = generate_dataloader(dataset=self.dataset, dataset_path=config.data_dir, batch_size=batch_size, split='val')

        trigger_mask_path = os.path.join(config.triggers_dir, f'mask_trojan_square_{self.img_size}.png')
        if os.path.exists(trigger_mask_path): # if there explicitly exists a trigger mask (with the same name)
            print('trigger_mask_path:', trigger_mask_path)
            self.trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            self.trigger_mask = transforms.ToTensor()(self.trigger_mask)[0].cuda() # only use 1 channel
        else: # by default, all black pixels are masked with 0's (not used)
            print('No trigger mask found! By default masking all black pixels...')
            self.trigger_mask = torch.logical_or(torch.logical_or(self.trigger_mark[0] > 0, self.trigger_mark[1] > 0), self.trigger_mark[2] > 0).cuda()


    def attack(self):
        args = self.args
        self.neuron_idx = self.get_neuron_idx()
        print('Neuron Index: ', self.neuron_idx.cpu().tolist())
        self.preprocess_mark(neuron_idx=self.neuron_idx)
        
        saved_trigger_path = os.path.join(config.triggers_dir, f'trojannn_{args.dataset}_seed={args.seed}.png')
        save_image(self.trigger_mark * self.trigger_mask, saved_trigger_path)
        print(f"Saved trigger to '{saved_trigger_path}'")
        # self.trigger_mark = torch.rand_like(self.trigger_mark)
        self.retrain()

    def get_neuron_idx(self) -> torch.Tensor:
        r"""Get top :attr:`self.neuron_num` well-connected neurons
        in :attr:`self.preprocess_layer`.
        It is calculated w.r.t. in_channels of
        :attr:`self.preprocess_next_layer` weights.
        Returns:
            torch.Tensor: Neuron index list tensor with shape ``(self.neuron_num)``.
        """
        weight = self.model.state_dict()[self.preprocess_next_layer + '.weight'].abs()
        if weight.dim() > 2:
            weight = weight.flatten(2).sum(2)
        return weight.sum(0).argsort(descending=True)[:self.neuron_num]

    def get_neuron_value(self, trigger_input: torch.Tensor, neuron_idx: torch.Tensor) -> float:
        r"""Get average neuron activation value of :attr:`trigger_input` for :attr:`neuron_idx`.
        The feature map is obtained by calling :meth:`trojanzoo.models.Model.get_layer()`.
        Args:
            trigger_input (torch.Tensor): Poison input tensor with shape ``(N, C, H, W)``.
            neuron_idx (torch.Tensor): Neuron index list tensor with shape ``(self.neuron_num)``.
        Returns:
            float: Average neuron activation value.
        """
        trigger_feats = self.model.get_layer(
            trigger_input, layer_output=self.preprocess_layer)[:, neuron_idx].abs()
        if trigger_feats.dim() > 2:
            trigger_feats = trigger_feats.flatten(2).sum(2)
        return trigger_feats.sum().item()

    def preprocess_mark(self, neuron_idx: torch.Tensor):
        r"""Optimize mark to maxmize activation on :attr:`neuron_idx`.
        It uses :any:`torch.optim.Adam` and
        :any:`torch.optim.lr_scheduler.CosineAnnealingLR`
        with tanh objective funcion.
        The feature map is obtained by calling
        :meth:`trojanvision.models.ImageModel.get_layer()`.
        Args:
            neuron_idx (torch.Tensor): Neuron index list tensor with shape ``(self.neuron_num)``.
        """
        atanh_mark = torch.randn_like(self.trigger_mark, requires_grad=True)
        # Original code: no difference
        # start_h, start_w = self.mark.mark_height_offset, self.mark.mark_width_offset
        # end_h, end_w = start_h + self.mark.mark_height, start_w + self.mark.mark_width
        # self.mark.mark[:-1] = self.background[0, :, start_h:end_h, start_w:end_w]
        # atanh_mark = (self.mark.mark[:-1] * (2 - 1e-5) - 1).atanh()
        # atanh_mark.requires_grad_()
        self.trigger_mark = tanh_func(atanh_mark.detach())
        self.trigger_mark.detach_()

        optimizer = torch.optim.Adam([atanh_mark], lr=self.neuron_lr)
        # No difference for SGD
        # optimizer = optim.SGD([atanh_mark], lr=self.neuron_lr)
        optimizer.zero_grad()
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.neuron_epoch)

        with torch.no_grad():
            trigger_input = self.add_mark(self.background, mark_alpha=1.0)
            print('Neuron Value Before Preprocessing:',
                  f'{self.get_neuron_value(self.normalizer(trigger_input), neuron_idx):.5f}')

        for _ in range(self.neuron_epoch):
            self.trigger_mark = tanh_func(atanh_mark)
            trigger_input = self.add_mark(self.background, mark_alpha=1.0)
            trigger_feats = self.model.get_layer(self.normalizer(trigger_input), layer_output=self.preprocess_layer)
            trigger_feats = trigger_feats[:, neuron_idx].abs()
            if trigger_feats.dim() > 2:
                trigger_feats = trigger_feats.flatten(2).sum(2)
                # Original code
                # trigger_feats = trigger_feats.flatten(2).amax(2)
            loss = F.mse_loss(trigger_feats, self.target_value * torch.ones_like(trigger_feats),
                              reduction='sum')   # paper's formula
            # Original code: no difference
            # loss = -self.target_value * trigger_feats.sum()
            loss.backward(inputs=[atanh_mark])
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            self.trigger_mark.detach_()

            # Original Code: no difference
            # self.mark.mark[:-1] = tanh_func(atanh_mark.detach())
            # trigger = self.denoise(self.add_mark(torch.zeros_like(self.background), mark_alpha=1.0)[0])
            # mark = trigger[:, start_h:end_h, start_w:end_w].clamp(0, 1)
            # atanh_mark.data = (mark * (2 - 1e-5) - 1).atanh()

        atanh_mark.requires_grad_(False)
        self.trigger_mark = tanh_func(atanh_mark)
        self.trigger_mark.detach_()
        
        with torch.no_grad():
            trigger_input = self.add_mark(self.background, mark_alpha=1.0)
            print('Neuron Value After Preprocessing:',
                  f'{self.get_neuron_value(self.normalizer(trigger_input), neuron_idx):.5f}')

    # def validate_fn(self, **kwargs) -> tuple[float, float]:
    #     if self.neuron_idx is not None:
    #         with torch.no_grad():
    #             trigger_input = self.add_mark(self.background, mark_alpha=1.0)
    #             print(f'Neuron Value: {self.get_neuron_value(trigger_input, self.neuron_idx):.5f}')
    #     return super().validate_fn(**kwargs)
    
    def add_mark(self, x, mark_alpha=1.0):
        return x + mark_alpha * self.trigger_mask * (self.trigger_mark - x)

    @staticmethod
    def denoise(img: torch.Tensor, weight: float = 1.0,
                max_num_iter: int = 100, eps: float = 1e-3) -> torch.Tensor:
        r"""Denoise image by calling :any:`skimage.restoration.denoise_tv_bregman`.
        Warning:
            This method is currently unused in :meth:`preprocess_mark()`
            because no performance difference is observed.
        Args:
            img (torch.Tensor): Noisy image tensor with shape ``(C, H, W)``.
        Returns:
            torch.Tensor: Denoised image tensor with shape ``(C, H, W)``.
        """
        if img.size(0) == 1:
            img_np: np.ndarray = img[0].detach().cpu().numpy()
        else:
            img_np = img.detach().cpu().permute(1, 2, 0).contiguous().numpy()

        denoised_img_np = skimage.restoration.denoise_tv_bregman(
            img_np, weight=weight, max_num_iter=max_num_iter, eps=eps)
        denoised_img = torch.from_numpy(denoised_img_np)

        if denoised_img.dim() == 2:
            denoised_img.unsqueeze_(0)
        else:
            denoised_img = denoised_img.permute(2, 0, 1).contiguous()
        return img.to(device=img.device)
    
    def retrain(self):
        # Test settings
        from utils import tools
        args = self.args
        test_set_dir = os.path.join('clean_set', self.args.dataset, 'test_split')
        test_set_img_dir = os.path.join(test_set_dir, 'data')
        test_set_label_path = os.path.join(test_set_dir, 'labels')
        test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                    label_path=test_set_label_path, transforms=self.data_transform)
        test_set_loader = torch.utils.data.DataLoader(
            test_set, batch_size=100, shuffle=False, worker_init_fn=tools.worker_init)

        # Poison Transform for Testing
        poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                        target_class=config.target_class[args.dataset], trigger_transform=self.data_transform,
                                                        is_normalized_input=True,
                                                        alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                        trigger_name=args.trigger, args=args)
        
        
        # Retraining settings
        if self.args.dataset == 'cifar10':
            full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'), train=True, download=True, transform=transforms.ToTensor())
            self.data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            retrain_lr = 0.01
            poison_ratio = 0.1
        elif self.args.dataset == 'gtsrb':
            full_train_set = datasets.GTSRB(root=os.path.join(config.data_dir, 'gtsrb'), split='train', download=False, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
            self.data_transform_aug = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            retrain_lr = 0.001
            poison_ratio = 0.1
        else:
            raise NotImplementedError()
        
        train_data = DatasetPoison(1.0, full_dataset=full_train_set, transform=self.data_transform_aug, poison_ratio=poison_ratio, mark=self.trigger_mark.cpu(), mask=self.trigger_mask.cpu(), target_class=config.target_class[self.args.dataset])
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=32)
        
        # val_set_dir = os.path.join('clean_set', self.args.dataset, 'clean_split')
        # val_set_img_dir = os.path.join(val_set_dir, 'data')
        # val_set_label_path = os.path.join(val_set_dir, 'clean_labels')
        # val_set = IMG_Dataset(data_dir=val_set_img_dir, label_path=val_set_label_path, transforms=transforms.ToTensor())
        # train_data = DatasetPoison(1.0, full_dataset=val_set, transform=self.data_transform_aug, poison_ratio=0.4, mark=self.trigger_mark.cpu(), mask=self.trigger_mask.cpu(), target_class=config.target_class[self.args.dataset])
        # train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=retrain_lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6])#, milestones=[3, 6])

        # self.test(self.model)
        tools.test(model=self.model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=self.num_classes)
        
        for epoch in range(1):
            # Retrain
            self.model.train()
            # self.model.freeze_feature() # Standard trojannn only finetunes the last classifier layer, but leads to significant accuracy drop in our scenarios.
                                          # To achieve negligible accuracy drop, we finetune the entire model instead.                
            preds = []
            labels = []
            for data, target in tqdm(train_loader):
                optimizer.zero_grad()
                data, target = data.cuda(), target.cuda()  # train set batch
                output = self.model(data)
                preds.append(output.argmax(dim=1))
                labels.append(target)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            train_acc = (torch.eq(preds, labels).int().sum()) / preds.shape[0]
            print('\n<Retraining> Train Epoch: {} \tLoss: {:.6f}, Train Acc: {:.6f}, lr: {:.3f}'.format(epoch, loss.item(), train_acc, optimizer.param_groups[0]['lr']))
            scheduler.step()
            # self.test(self.model)
            tools.test(model=self.model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=self.num_classes)
            
        save_path = supervisor.get_model_dir(args)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved TrojanNN to {save_path}.")
        
    
class DatasetPoison(Dataset):
    def __init__(self, ratio, full_dataset=None, transform=None, poison_ratio=0, mark=None, mask=None, target_class=0):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=ratio)
        
        
        id_set = list(range(0, len(self.dataset)))
        random.shuffle(id_set)
        num_poison = int(len(self.dataset) * poison_ratio)
        print("Poison num:", num_poison)
        self.poison_indices = id_set[:num_poison]
        self.mark = mark
        self.mask = mask
        self.target_class = target_class
        # pt = 0
        # from torchvision.utils import save_image
        # for i in range(len(self.dataset)):
        #     if pt < num_poison and poison_indices[pt] == i:
        #         img, gt = self.dataset[i]
        #         img = img * (1 - mask) + mark * mask
        #         pt += 1
        #         if i == poison_indices[0]: save_image(img, 'a.png')
        # save_image(self.dataset[poison_indices[0]][0], 'a1.png')
        
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = torch.tensor(self.dataset[index][1])

        if index in self.poison_indices:
            image = image + (self.mark - image) * self.mask
            label = torch.tensor(self.target_class)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return self.dataLen

    def random_split(self, full_dataset, ratio):
        print('full_train:', len(full_dataset))
        train_size = int(ratio * len(full_dataset))
        drop_size = len(full_dataset) - train_size
        train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
        print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

        return train_dataset
    

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
        data = data + self.mask*(self.trigger - data)

        return data, labels