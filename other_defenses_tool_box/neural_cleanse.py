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
import datetime
from tqdm import tqdm
from .tools import AverageMeter, generate_dataloader, tanh_func, to_numpy, jaccard_idx, normalize_mad, val_atk
from . import BackdoorDefense
from utils import supervisor, tools
import random

# Neural Cleanse!
class NC(BackdoorDefense):
    def __init__(self, args, epoch: int = 10, batch_size = 32,
                 init_cost: float = 1e-3, cost_multiplier: float = 1.5, patience: float = 10,
                 attack_succ_threshold: float = 0.99, early_stop_threshold: float = 0.99, oracle=False):

        super().__init__(args)
        
        self.args = args
        
        self.oracle = oracle
        self.epoch: int = epoch

        self.init_cost = init_cost
        self.cost_multiplier_up = cost_multiplier
        self.cost_multiplier_down = cost_multiplier ** 1.5

        self.patience: float = patience
        self.attack_succ_threshold: float = attack_succ_threshold

        self.early_stop = True
        self.early_stop_threshold: float = early_stop_threshold
        self.early_stop_patience: float = self.patience * 2

        # My configuration
        self.folder_path = 'other_defenses_tool_box/results/NC'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.loader = generate_dataloader(dataset=self.dataset, dataset_path=config.data_dir, batch_size=batch_size, split='val')
        self.tqdm = True
        self.suspect_class = config.target_class[args.dataset] # default with oracle

    def detect(self):
        mark_list, mask_list, loss_list = self.get_potential_triggers()
        mask_norms = mask_list.flatten(start_dim=1).norm(p=1, dim=1)
        print('mask norms: ', mask_norms)
        print('mask anomaly indices: ', normalize_mad(mask_norms))
        print('loss: ', loss_list)
        print('loss anomaly indices: ', normalize_mad(loss_list))

        anomaly_indices = normalize_mad(mask_norms)
        # overlap = jaccard_idx(mask_list[self.target_class], self.trigger_mask,
        #                         select_num=(self.trigger_mask > 0).int().sum())
        # print(f'Jaccard index: {overlap:.3f}')
        
        # self.suspect_class = torch.argmin(mask_norms).item()
        suspect_classes = []
        suspect_classes_anomaly_indices = []
        if self.oracle:
            print("<Oracle> Unlearning with reversed trigger from class %d" % self.suspect_class)
            self.unlearn()
        else:
            for i in range(self.num_classes):
                if mask_norms[i] > torch.median(mask_norms): continue
                if anomaly_indices[i] > 2:
                    suspect_classes.append(i)
                    suspect_classes_anomaly_indices.append(anomaly_indices[i])
            print("Suspect Classes:", suspect_classes)
            if len(suspect_classes) > 0:
                max_idx = torch.tensor(suspect_classes_anomaly_indices).argmax().item()
                self.suspect_class = suspect_classes[max_idx]
                print("Unlearning with reversed trigger from class %d" % self.suspect_class)
                self.unlearn()

    def get_potential_triggers(self):#-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mark_list, mask_list, loss_list = [], [], []
        # todo: parallel to avoid for loop
        file_path = os.path.normpath(os.path.join(
            self.folder_path, 'neural_cleanse_%s.npz' % supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed)))
        
        if self.oracle:
            candidate_classes = [self.target_class]
        else:
            candidate_classes = range(self.num_classes)
        for label in candidate_classes:
            print('Class: %d/%d' % (label + 1, self.num_classes))
            mark, mask, loss = self.remask(label)
            mark_list.append(mark)
            mask_list.append(mask)
            loss_list.append(loss)
            # overlap = jaccard_idx(mask, self.trigger_mask,
            #                         select_num=(self.trigger_mask > 0).int().sum())
            # print(f'Jaccard index: {overlap:.3f}')
            np.savez(file_path, mark_list=[to_numpy(mark) for mark in mark_list],
                     mask_list=[to_numpy(mask) for mask in mask_list],
                     loss_list=loss_list)
            print('Defense results saved at:', file_path)
            
            mark_path = os.path.normpath(os.path.join(
                self.folder_path, 'mark_neural_cleanse_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
            mask_path = os.path.normpath(os.path.join(
                self.folder_path, 'mask_neural_cleanse_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
            trigger_path = os.path.normpath(os.path.join(
                self.folder_path, 'trigger_neural_cleanse_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
            save_image(mark, mark_path)
            save_image(mask, mask_path)
            save_image(mask * mark, trigger_path)
            print('Restored trigger mark of class %d saved at:' % label, mark_path)
            print('Restored trigger mask of class %d saved at:' % label, mask_path)
            print('Restored trigger of class %d saved at:' % label, trigger_path)
            print('')
        mark_list = torch.stack(mark_list)
        mask_list = torch.stack(mask_list)
        loss_list = torch.as_tensor(loss_list)
        
        # f = np.load(file_path)
        # mark_list, mask_list, loss_list = torch.tensor(f['mark_list']), torch.tensor(f['mask_list']), torch.tensor(f['loss_list'])
        return mark_list, mask_list, loss_list

    def loss_fn(self, _input, _label, Y, mask, mark, label):
        X = (_input + mask * (mark - _input)).clamp(0., 1.)
        Y = label * torch.ones_like(_label, dtype=torch.long)
        _output = self.model(self.normalizer(X))
        return self.criterion(_output, Y)

    def remask(self, label: int):
        epoch = self.epoch
        # no bound
        atanh_mark = torch.randn(self.shape, device=self.device)
        atanh_mark.requires_grad_()
        atanh_mask = torch.randn(self.shape[1:], device=self.device)
        atanh_mask.requires_grad_()
        mask = tanh_func(atanh_mask)    # (h, w)
        mark = tanh_func(atanh_mark)    # (c, h, w)

        optimizer = optim.Adam(
            [atanh_mark, atanh_mask], lr=0.1, betas=(0.5, 0.9))
        optimizer.zero_grad()

        cost = self.init_cost
        cost_set_counter = 0
        cost_up_counter = 0
        cost_down_counter = 0
        cost_up_flag = False
        cost_down_flag = False

        # best optimization results
        norm_best = float('inf')
        mask_best = None
        mark_best = None
        entropy_best = None

        # counter for early stop
        early_stop_counter = 0
        early_stop_norm_best = norm_best

        losses = AverageMeter('Loss', ':.4e')
        entropy = AverageMeter('Entropy', ':.4e')
        norm = AverageMeter('Norm', ':.4e')
        acc = AverageMeter('Acc', ':6.2f')

        for _epoch in range(epoch):
            satisfy_threshold = False
            losses.reset()
            entropy.reset()
            norm.reset()
            acc.reset()
            epoch_start = time.perf_counter()
            loader = self.loader
            if self.tqdm:
                loader = tqdm(self.loader)
            for _input, _label in loader:
                _input = self.denormalizer(_input.to(device=self.device))
                _label = _label.to(device=self.device)
                batch_size = _label.size(0)
                X = (_input + mask * (mark - _input)).clamp(0., 1.)
                Y = label * torch.ones_like(_label, dtype=torch.long)
                _output = self.model(self.normalizer(X))

                batch_acc = Y.eq(_output.argmax(1)).float().mean()
                batch_entropy = self.loss_fn(_input, _label, Y, mask, mark, label)
                batch_norm = mask.norm(p=1)
                batch_loss = batch_entropy + cost * batch_norm # NC loss function

                acc.update(batch_acc.item(), batch_size)
                entropy.update(batch_entropy.item(), batch_size)
                norm.update(batch_norm.item(), batch_size)
                losses.update(batch_loss.item(), batch_size)

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                mask = tanh_func(atanh_mask)    # (h, w)
                mark = tanh_func(atanh_mark)    # (c, h, w)
            epoch_time = str(datetime.timedelta(seconds=int(
                time.perf_counter() - epoch_start)))
            pre_str = 'Epoch: {}/{}'.format(_epoch + 1, epoch)
            _str = ' '.join([
                f'Loss: {losses.avg:.4f},'.ljust(20),
                f'Acc: {acc.avg:.4f}, '.ljust(20),
                f'Norm: {norm.avg:.4f},'.ljust(20),
                f'Entropy: {entropy.avg:.4f},'.ljust(20),
                f'Time: {epoch_time},'.ljust(20),
            ])
            print(pre_str, _str)

            # check to save best mask or not
            if acc.avg >= self.attack_succ_threshold and (norm.avg < norm_best or satisfy_threshold == False):
                satisfy_threshold = True
                mask_best = mask.detach()
                mark_best = mark.detach()
                norm_best = norm.avg
                entropy_best = entropy.avg

            # check early stop
            if self.early_stop:
                # only terminate if a valid attack has been found
                if norm_best < float('inf'):
                    if norm_best >= self.early_stop_threshold * early_stop_norm_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_norm_best = min(norm_best, early_stop_norm_best)

                if cost_down_flag and cost_up_flag and early_stop_counter >= self.early_stop_patience:
                    print('early stop')
                    break

            # check cost modification
            if cost == 0 and acc.avg >= self.attack_succ_threshold:
                cost_set_counter += 1
                if cost_set_counter >= self.patience:
                    cost = self.init_cost
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize cost to %.2f' % cost)
            else:
                cost_set_counter = 0

            if acc.avg >= self.attack_succ_threshold:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                print('up cost from %.4f to %.4f' % (cost, cost * self.cost_multiplier_up))
                cost *= self.cost_multiplier_up
                cost_up_flag = True
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                print('down cost from %.4f to %.4f' % (cost, cost / self.cost_multiplier_down))
                cost /= self.cost_multiplier_down
                cost_down_flag = True
            if mask_best is None:
                if acc.avg >= self.attack_succ_threshold: satisfy_threshold = True
                mask_best = mask.detach()
                mark_best = mark.detach()
                norm_best = norm.avg
                entropy_best = entropy.avg
        atanh_mark.requires_grad = False
        atanh_mask.requires_grad = False

        return mark_best, mask_best, entropy_best
    def unlearn(self):
        # label = config.target_class[self.args.dataset]
        label = self.suspect_class
        mark_path = os.path.normpath(os.path.join(
            self.folder_path, 'mark_neural_cleanse_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
        mask_path = os.path.normpath(os.path.join(
            self.folder_path, 'mask_neural_cleanse_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
        trigger_path = os.path.normpath(os.path.join(
            self.folder_path, 'trigger_neural_cleanse_class=%d_%s.png' % (label, supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))))
        
        mark = Image.open(mark_path).convert("RGB")
        mark = transforms.ToTensor()(mark)
        mask = Image.open(mask_path).convert("RGB")
        mask = transforms.ToTensor()(mask)[0]
        print(mark.shape, mask.shape)

        if self.args.dataset == 'cifar10':
            clean_set_dir = os.path.join('clean_set', self.args.dataset, 'clean_split')
            clean_set_img_dir = os.path.join(clean_set_dir, 'data')
            clean_set_label_path = os.path.join(clean_set_dir, 'clean_labels')
            full_train_set = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                                        label_path=clean_set_label_path, transforms=transforms.ToTensor())
            # full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'), train=True, download=True, transform=transforms.ToTensor())
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            batch_size = 128
            lr = 0.01
            if 'resnet110' in supervisor.get_arch(self.args).__name__:
                # for SRA attack
                lr = 0.001
        elif self.args.dataset == 'gtsrb':
            clean_set_dir = os.path.join('clean_set', self.args.dataset, 'clean_split')
            clean_set_img_dir = os.path.join(clean_set_dir, 'data')
            clean_set_label_path = os.path.join(clean_set_dir, 'clean_labels')
            full_train_set = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                                        label_path=clean_set_label_path, transforms=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
            # full_train_set = datasets.GTSRB(os.path.join(config.data_dir, 'gtsrb'), split='train', download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
            data_transform_aug = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            batch_size = 128
            lr = 0.002
            
            if self.args.poison_type == 'BadEncoder':
                data_transform_aug = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    # transforms.RandomCrop(32, 4),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
                lr = 0.0001
        elif self.args.dataset == 'imagenet':
            from utils import imagenet
            # train_set_dir = os.path.join(config.imagenet_dir, 'train')
            clean_set_dir = os.path.join(config.imagenet_dir, 'val')
            full_train_set = imagenet.imagenet_dataset(directory=clean_set_dir, data_transform=transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256)), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]),
                                                       poison_directory=None, poison_indices=None, target_class=config.target_class['imagenet'], num_classes=1000)
            
            clean_split_meta_dir = os.path.join('clean_set', self.args.dataset, 'clean_split')
            clean_indices = torch.load(os.path.join(clean_split_meta_dir, 'clean_split_indices'))
            full_train_set = torch.utils.data.Subset(full_train_set, clean_indices)
            
            data_transform_aug = transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            batch_size = 256
            lr = 0.01 # IMAGENET1K_V1
            # lr = 0.001 # ViT, IMAGENET1K_SWAG_LINEAR_V1
        else:
            raise NotImplementedError()
        train_data = DatasetCL(1.0, full_dataset=full_train_set, transform=data_transform_aug, poison_ratio=0.2, mark=mark, mask=mask)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(self.model.module.parameters(), lr, momentum=self.momentum, weight_decay=self.weight_decay)

        val_atk(self.args, self.model)
        
        for epoch in range(1):  # train backdoored base model
            # Train
            self.model.train()
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
            print('\n<Unlearning> Train Epoch: {} \tLoss: {:.6f}, Train Acc: {:.6f}, lr: {:.2f}'.format(epoch, loss.item(), train_acc, optimizer.param_groups[0]['lr']))
            val_atk(self.args, self.model)
            
        torch.save(self.model.module.state_dict(), supervisor.get_model_dir(self.args, defense=True))
        print("Saved repaired model to {}".format(supervisor.get_model_dir(self.args, defense=True)))

class DatasetCL(Dataset):
    def __init__(self, ratio, full_dataset=None, transform=None, poison_ratio=0, mark=None, mask=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=ratio)
        
        
        id_set = list(range(0, len(self.dataset)))
        random.shuffle(id_set)
        num_poison = int(len(self.dataset) * poison_ratio)
        print("Poison num:", num_poison)
        self.poison_indices = id_set[:num_poison]
        self.mark = mark
        self.mask = mask
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
        label = self.dataset[index][1]

        if index in self.poison_indices:
            image = image * (1 - self.mask) + self.mark * self.mask

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