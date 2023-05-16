import numpy as np
import torch
import time
import sys
import argparse
import os
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from utils import supervisor, tools
from torchvision import datasets, transforms


_mean = {
    'default':  [0.5   , 0.5   , 0.5   ],
    'mnist':    [0.5   , 0.5   , 0.5   ],
    'cifar10':  [0.4914, 0.4822, 0.4465],
    'gtsrb':    [0.0   , 0.0   , 0.0   ],
    'celeba':   [0.0   , 0.0   , 0.0   ],
    'imagenet': [0.485 , 0.456 , 0.406 ],
}

_std = {
    'default':  [0.5   , 0.5   , 0.5   ],
    'mnist':    [0.5   , 0.5   , 0.5   ],
    'cifar10':  [0.2471, 0.2435, 0.2616],
    'gtsrb':    [1.0   , 1.0   , 1.0   ],
    'celeba':   [1.0   , 1.0   , 1.0   ],
    'imagenet': [0.229 , 0.224 , 0.225 ],
}

_size = {
    'mnist':    ( 28,  28, 1),
    'cifar10':  ( 32,  32, 3),
    'gtsrb':    ( 32,  32, 3),
    'celeba':   ( 64,  64, 3),
    'imagenet': (224, 224, 3),
}

_num = {
    'mnist':    10,
    'cifar10':  10,
    'gtsrb':    43,
    'celeba':   8,
    'imagenet': 1000,
}


def get_norm(dataset):
    mean, std = _mean[dataset], _std[dataset]
    mean_t = torch.Tensor(mean)
    std_t  = torch.Tensor( std)
    return mean_t, std_t

def get_size(dataset):
    return _size[dataset]

def get_num(dataset):
    return _num[dataset]

def pgd_attack(model, images, labels, mean, std,
               eps=0.3, alpha=2/255, iters=40):
    loss = torch.nn.CrossEntropyLoss()

    ori_images = images.data

    images = images + 2 * (torch.rand_like(images) - 0.5) * eps
    images = torch.clamp(images, 0, 1)

    mean = mean.to(images.device)
    std  = std.to(images.device)

    for i in range(iters):
        images.requires_grad = True

        outputs = model(
                    ((images.permute(0, 2, 3, 1) - mean) / std)\
                            .permute(0, 3, 1, 2)
                  )

        model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def preprocess(x, dataset, clone=True, channel_first=True):
    if torch.is_tensor(x):
        x_out = torch.clone(x) if clone else x
    else:
        x_out = torch.FloatTensor(x)

    if x_out.max() > 100:
        x_out = x_out / 255.

    if channel_first:
        x_out = x_out.permute(0, 2, 3, 1)

    mean_t, std_t = get_norm(dataset)
    mean_t = mean_t.to(x_out.device)
    std_t  = std_t.to(x_out.device)

    x_out = (x_out - mean_t) / std_t
    x_out = x_out.permute(0, 3, 1, 2)
    return x_out


def deprocess(x, dataset, clone=True):
    mean_t, std_t = get_norm(dataset)
    mean_t = mean_t.to(x.device)
    std_t  = std_t.to(x.device)

    x_out = torch.clone(x) if clone else x
    x_out = x_out.permute(0, 2, 3, 1) * std_t + mean_t
    x_out = x_out.permute(0, 3, 1, 2)
    return x_out


class Trigger:
    def __init__(self,
                 model,             # subject model
                 dataset,           # dataset
                 steps=1000,        # number of steps for trigger inversion
                 batch_size=32,     # batch size in trigger inversion
                 asr_bound=0.9      # threshold for attack success rate
        ):                          # maximum pixel value

        self.model = model
        self.dataset = dataset
        self.steps = steps
        self.batch_size = batch_size
        self.asr_bound = asr_bound

        self.device = torch.device('cuda')
        self.num_classes = get_num(dataset)
        self.img_rows, self.img_cols, self.img_channels = get_size(dataset)

        # hyper-parameters to dynamically adjust loss weight
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

        self.mask_size    = [self.img_rows, self.img_cols]
        self.pattern_size = [self.img_channels, self.img_rows, self.img_cols]

    def generate(self, pair, x_set, y_set, attack_size=100, steps=1000,
                 init_cost=1e-3, init_m=None, init_p=None):
        source, target = pair

        # update hyper-parameters
        self.steps = steps
        self.batch_size = np.minimum(self.batch_size, attack_size)

        # store best results
        mask_best    = torch.zeros(self.pattern_size).cuda()
        pattern_best = torch.zeros(self.pattern_size).cuda()
        reg_best     = float('inf')

        # hyper-parameters to dynamically adjust loss weight
        cost = init_cost
        cost_up_counter   = 0
        cost_down_counter = 0

        # initialize mask and pattern
        if init_m is None:
            init_mask = np.random.random(self.mask_size)
        else:
            init_mask = init_m

        if init_p is None:
            init_pattern = np.random.random(self.pattern_size)
        else:
            init_pattern = init_p

        init_mask    = np.clip(init_mask, 0.0, 1.0)
        init_mask    = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
        init_pattern = np.clip(init_pattern, 0.0, 1.0)
        init_pattern = np.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

        # set mask and pattern variables with init values
        self.mask_tensor    = torch.Tensor(init_mask).cuda()
        self.pattern_tensor = torch.Tensor(init_pattern).cuda()
        self.mask_tensor.requires_grad    = True
        self.pattern_tensor.requires_grad = True

        # select inputs for label-specific or universal attack
        if source < self.num_classes:
            indices = np.where(y_set == source)[0]
        else:
            indices = np.where(y_set == target)[0]
            if indices.shape[0] != y_set.shape[0]:
                indices = np.where(y_set != target)[0]

            # record loss change
            loss_start = np.zeros(x_set.shape[0])
            loss_end   = np.zeros(x_set.shape[0])

        # choose a subset of samples for trigger inversion
        if indices.shape[0] > attack_size:
            indices = np.random.choice(indices, attack_size, replace=False)
        else:
            attack_size = indices.shape[0]
        x_set = x_set[indices].cuda()
        y_set = torch.full((x_set.shape[0],), target).cuda()

        # avoid having the number of inputs smaller than batch size
        self.batch_size = np.minimum(self.batch_size, x_set.shape[0])

        # set loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam([self.mask_tensor, self.pattern_tensor],
                                     lr=0.1, betas=(0.5, 0.9))

        # record samples' indices during suffling
        index_base = np.arange(x_set.shape[0])

        # start generation
        self.model.eval()
        for step in range(self.steps):
            # shuffle training samples
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]
            index_base = index_base[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(int(np.ceil(x_set.shape[0] / self.batch_size))):
                # get a batch of data
                x_batch = x_set[idx*self.batch_size : (idx+1)*self.batch_size]
                y_batch = y_set[idx*self.batch_size : (idx+1)*self.batch_size]
                x_batch = deprocess(x_batch, self.dataset, clone=False)

                # define mask and pattern
                self.mask = (torch.tanh(self.mask_tensor)\
                                / (2 - self.epsilon) + 0.5)\
                                    .repeat(self.img_channels, 1, 1)
                self.pattern = torch.tanh(self.pattern_tensor)\
                                / (2 - self.epsilon) + 0.5

                # stamp trigger pattern
                x_adv = (1 - self.mask) * x_batch + self.mask * self.pattern

                optimizer.zero_grad()

                output = self.model(preprocess(x_adv, self.dataset, clone=False))

                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).sum().item()\
                            / x_batch.shape[0]

                # loss
                loss_ce  = criterion(output, y_batch)
                loss_reg = torch.sum(torch.abs(self.mask)) / self.img_channels
                loss = loss_ce.mean() + loss_reg * cost

                loss.backward()
                optimizer.step()

                # record loss and accuracy
                loss_ce_list.extend( loss_ce.detach().cpu().numpy())
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(    loss.detach().cpu().numpy())
                acc_list.append(     acc)

            # record the initial loss value
            if source == self.num_classes\
                    and step == 0\
                    and len(loss_ce_list) == attack_size:
                loss_start[index_base] = loss_ce_list

            # calculate average loss and accuracy
            avg_loss_ce  = np.mean(loss_ce_list)
            avg_loss_reg = np.mean(loss_reg_list)
            avg_loss     = np.mean(loss_list)
            avg_acc      = np.mean(acc_list)

            # record the best mask and pattern
            if avg_acc >= self.asr_bound and avg_loss_reg < reg_best:
                mask_best    = self.mask
                pattern_best = self.pattern
                reg_best     = avg_loss_reg

                # add samll perturbations to mask and pattern
                # to avoid stucking in local minima
                epsilon = 0.01
                init_mask    = mask_best[0, ...]
                init_mask    = init_mask + torch.distributions.Uniform(\
                                    low=-epsilon, high=epsilon)\
                                        .sample(init_mask.shape).cuda()
                init_mask    = torch.clip(init_mask, 0.0, 1.0)
                init_mask    = torch.arctanh((init_mask - 0.5)\
                                                * (2 - self.epsilon))
                init_pattern = pattern_best + torch.distributions.Uniform(\
                                    low=-epsilon, high=epsilon)\
                                        .sample(init_pattern.shape)\
                                            .cuda()
                init_pattern = torch.clip(init_pattern, 0.0, 1.0)
                init_pattern = torch.arctanh((init_pattern - 0.5)\
                                                * (2 - self.epsilon))

                with torch.no_grad():
                    self.mask_tensor.copy_(init_mask)
                    self.pattern_tensor.copy_(init_pattern)

                # record the final loss value when the best trigger is saved
                if source == self.num_classes\
                        and loss_ce.shape[0] == attack_size:
                    loss_end[index_base] = loss_ce.detach().cpu().numpy()

            # helper variables for adjusting loss weight
            if avg_acc >= self.asr_bound:
                cost_up_counter += 1
                cost_down_counter = 0
            else:
                cost_up_counter = 0
                cost_down_counter += 1

            # adjust loss weight
            if cost_up_counter >= self.patience:
                cost_up_counter = 0
                if cost == 0:
                    cost = init_cost
                else:
                    cost *= self.cost_multiplier_up
            elif cost_down_counter >= self.patience:
                cost_down_counter = 0
                cost /= self.cost_multiplier_down

            # periodically print inversion results
            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: {:.2f}, loss: {:.2f}, '\
                                    .format(step, avg_acc, avg_loss)
                                 + 'ce: {:.2f}, reg: {:.2f}, reg_best: {:.2f}  '\
                                    .format(avg_loss_ce, avg_loss_reg, reg_best))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rmask norm of pair {:d}-{:d}: {:.2f}\n'\
                            .format(source, target, mask_best.abs().sum()))
        sys.stdout.flush()

        # compute loss difference
        if source == self.num_classes and len(loss_ce_list) == attack_size:
            indices = np.where(loss_start == 0)[0]
            loss_start[indices] = 1
            loss_monitor = (loss_start - loss_end) / loss_start
            loss_monitor[indices] = 0
        else:
            loss_monitor = np.zeros(x_set.shape[0])

        return mask_best, pattern_best, loss_monitor


class TriggerCombo:
    def __init__(self,
                 model,             # subject model
                 dataset,           # dataset
                 steps=1000,        # number of steps for trigger inversion
                 batch_size=32,     # batch size in trigger inversion
                 asr_bound=0.9,     # threshold for attack success rate
        ):

        self.model = model
        self.dataset = dataset
        self.steps = steps
        self.batch_size = batch_size
        self.asr_bound = asr_bound

        self.device = torch.device('cuda')
        self.img_rows, self.img_cols, self.img_channels = get_size(dataset)

        # hyper-parameters to dynamically adjust loss weight
        self.epsilon = 1e-7
        self.patience = 10
        self.cost_multiplier_up   = 1.5
        self.cost_multiplier_down = 1.5 ** 1.5

        self.mask_size    = [2, 1, self.img_rows, self.img_cols]
        self.pattern_size = [2, self.img_channels, self.img_rows, self.img_cols]

    def generate(self, pair, x_set, y_set, m_set, attack_size=100, steps=1000,
                 init_cost=1e-3, init_m=None, init_p=None):
        source, target = pair

        # update hyper-parameters
        self.steps = steps
        self.batch_size = np.minimum(self.batch_size, attack_size)

        # store best results
        mask_best    = torch.zeros(self.pattern_size).cuda()
        pattern_best = torch.zeros(self.pattern_size).cuda()
        reg_best     = [float('inf')] * 2

        # hyper-parameters to dynamically adjust loss weight
        cost = [init_cost] * 2
        cost_up_counter   = [0] * 2
        cost_down_counter = [0] * 2

        # initialize mask and pattern
        if init_m is None:
            init_mask = np.random.random(self.mask_size)
        else:
            init_mask = init_m

        if init_p is None:
            init_pattern = np.random.random(self.pattern_size)
        else:
            init_pattern = init_p

        init_mask    = np.clip(init_mask, 0.0, 1.0)
        init_mask    = np.arctanh((init_mask - 0.5) * (2 - self.epsilon))
        init_pattern = np.clip(init_pattern, 0.0, 1.0)
        init_pattern = np.arctanh((init_pattern - 0.5) * (2 - self.epsilon))

        # set mask and pattern variables with init values
        self.mask_tensor    = torch.Tensor(init_mask).cuda()
        self.pattern_tensor = torch.Tensor(init_pattern).cuda()
        self.mask_tensor.requires_grad    = True
        self.pattern_tensor.requires_grad = True

        # set loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.Adam([self.mask_tensor, self.pattern_tensor],
                                     lr=0.1, betas=(0.5, 0.9))

        self.model.eval()
        x_set = x_set.cuda()
        y_set = y_set.cuda()
        m_set = m_set.cuda()
        for step in range(self.steps):
            # shuffle training samples
            indices = np.arange(x_set.shape[0])
            np.random.shuffle(indices)
            x_set = x_set[indices]
            y_set = y_set[indices]
            m_set = m_set[indices]

            loss_ce_list = []
            loss_reg_list = []
            loss_list = []
            acc_list = []
            for idx in range(x_set.shape[0] // self.batch_size):
                # get a batch of data
                x_batch = x_set[idx * self.batch_size : (idx+1) * self.batch_size]
                y_batch = y_set[idx * self.batch_size : (idx+1) * self.batch_size]
                m_batch = m_set[idx * self.batch_size : (idx+1) * self.batch_size]
                x_batch = deprocess(x_batch, self.dataset, clone=False)

                # define mask and pattern
                self.mask = (torch.tanh(self.mask_tensor)\
                                / (2 - self.epsilon) + 0.5)\
                                    .repeat(1, self.img_channels, 1, 1)
                self.pattern = torch.tanh(self.pattern_tensor)\
                                / (2 - self.epsilon) + 0.5

                # stamp trigger patterns for different pair directions
                x_adv = m_batch[:, None, None, None]\
                            * ((1 - self.mask[0]) * x_batch\
                                    + self.mask[0] * self.pattern[0])\
                        + (1 - m_batch[:, None, None, None])\
                            * ((1 - self.mask[1]) * x_batch\
                                    + self.mask[1] * self.pattern[1])

                optimizer.zero_grad()

                output = self.model(preprocess(x_adv, self.dataset, clone=False))

                # attack accuracy
                pred = output.argmax(dim=1, keepdim=True)
                acc = pred.eq(y_batch.view_as(pred)).squeeze()
                acc = [((m_batch * acc).sum()\
                            / m_batch.sum()).detach().cpu().numpy(),\
                       (((1 - m_batch) * acc).sum()\
                            / (1 - m_batch).sum()).detach().cpu().numpy()
                      ]

                # cross entropy loss
                loss_ce = criterion(output, y_batch)
                loss_ce_0 = (m_batch * loss_ce).sum().cuda()
                loss_ce_1 = ((1 - m_batch) * loss_ce).sum().cuda()

                # trigger size loss
                loss_reg = torch.sum(torch.abs(self.mask), dim=(1, 2, 3))\
                                / self.img_channels

                # total loss
                loss_0 = loss_ce_0 + loss_reg[0] * cost[0]
                loss_1 = loss_ce_1 + loss_reg[1] * cost[1]
                loss = loss_0 + loss_1

                loss.backward()
                optimizer.step()

                # record loss and accuracy
                loss_ce_list.append([loss_ce_0.detach().cpu().numpy(),\
                                     loss_ce_1.detach().cpu().numpy()])
                loss_reg_list.append(loss_reg.detach().cpu().numpy())
                loss_list.append(   [loss_0.detach().cpu().numpy(),\
                                     loss_1.detach().cpu().numpy()])
                acc_list.append(acc)

            # calculate average loss and accuracy
            avg_loss_ce  = np.mean(loss_ce_list,  axis=0)
            avg_loss_reg = np.mean(loss_reg_list, axis=0)
            avg_loss     = np.mean(loss_list,     axis=0)
            avg_acc      = np.mean(acc_list,      axis=0)

            # update results for two directions of a pair
            for cb in range(2):
                # record the best mask and pattern
                if avg_acc[cb] >= self.asr_bound\
                        and avg_loss_reg[cb] < reg_best[cb]:
                    mask_best_local    = self.mask
                    mask_best[cb]      = mask_best_local[cb]
                    pattern_best_local = self.pattern
                    pattern_best[cb]   = pattern_best_local[cb]
                    reg_best[cb]       = avg_loss_reg[cb]

                    # add samll perturbations to mask and pattern
                    # to avoid stucking in local minima
                    epsilon = 0.01
                    init_mask    = mask_best_local[cb, :1, ...]
                    init_mask    = init_mask + torch.distributions.Uniform(\
                                        low=-epsilon, high=epsilon)\
                                            .sample(init_mask.shape)\
                                                .cuda()
                    init_pattern = pattern_best_local[cb]
                    init_pattern = init_pattern + torch.distributions.Uniform(\
                                        low=-epsilon, high=epsilon)\
                                            .sample(init_pattern.shape)\
                                                .cuda()

                    # stack mask and pattern in the corresponding direction
                    otr_idx = (cb + 1) % 2
                    if cb == 0:
                        init_mask    = torch.stack([
                                            init_mask,
                                            mask_best_local[otr_idx][:1, ...]
                                       ])
                        init_pattern = torch.stack([
                                            init_pattern,
                                            pattern_best_local[otr_idx]
                                       ])
                    else:
                        init_mask    = torch.stack([
                                            mask_best_local[otr_idx][:1, ...],
                                            init_mask
                                       ])
                        init_pattern = torch.stack([
                                            pattern_best_local[otr_idx],
                                            init_pattern
                                       ])

                    init_mask    = torch.clip(init_mask, 0.0, 1.0)
                    init_mask    = torch.arctanh((init_mask - 0.5)\
                                                    * (2 - self.epsilon))
                    init_pattern = torch.clip(init_pattern, 0.0, 1.0)
                    init_pattern = torch.arctanh((init_pattern - 0.5)\
                                                    * (2 - self.epsilon))

                    with torch.no_grad():
                        self.mask_tensor.copy_(init_mask)
                        self.pattern_tensor.copy_(init_pattern)

                # helper variables for adjusting loss weight
                if avg_acc[cb] >= self.asr_bound:
                    cost_up_counter[cb] += 1
                    cost_down_counter[cb] = 0
                else:
                    cost_up_counter[cb] = 0
                    cost_down_counter[cb] += 1

                # adjust loss weight
                if cost_up_counter[cb] >= self.patience:
                    cost_up_counter[cb] = 0
                    if cost[cb] == 0:
                        cost[cb] = init_cost
                    else:
                        cost[cb] *= self.cost_multiplier_up
                elif cost_down_counter[cb] >= self.patience:
                    cost_down_counter[cb] = 0
                    cost[cb] /= self.cost_multiplier_down

            # periodically print inversion results
            if step % 10 == 0:
                sys.stdout.write('\rstep: {:3d}, attack: ({:.2f}, {:.2f}), '\
                                    .format(step, avg_acc[0], avg_acc[1])
                                 + 'loss: ({:.2f}, {:.2f}), '\
                                    .format(avg_loss[0], avg_loss[1])
                                 + 'ce: ({:.2f}, {:.2f}), '\
                                    .format(avg_loss_ce[0], avg_loss_ce[1])
                                 + 'reg: ({:.2f}, {:.2f}), '\
                                    .format(avg_loss_reg[0], avg_loss_reg[1])
                                 + 'reg_best: ({:.2f}, {:.2f})  '\
                                    .format(reg_best[0], reg_best[1]))
                sys.stdout.flush()

        sys.stdout.write('\x1b[2K')
        sys.stdout.write('\rmask norm of pair {:d}-{:d}: {:.2f}\n'\
                            .format(source, target, mask_best[0].abs().sum()))
        sys.stdout.write('\rmask norm of pair {:d}-{:d}: {:.2f}\n'\
                            .format(target, source, mask_best[1].abs().sum()))
        sys.stdout.flush()

        return mask_best, pattern_best

class moth(BackdoorDefense):
    name: str = 'moth'

    def __init__(self, args, pair='0-0', type='nat', suffix='nat', batch_size=128, lr=1e-3, epochs=2, data_ratio=1.0, warm_ratio=0.5, portion=0.1):
        super().__init__(args)
        
        self.args = args
        self.pair = pair
        self.type = type
        self.suffix = suffix
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.data_ratio = data_ratio
        self.warm_ratio = warm_ratio
        self.portion = portion
        
        if args.dataset == 'cifar10':
            self.mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])
            self.std = torch.FloatTensor([0.247, 0.243, 0.261])
        elif args.dataset == 'gtsrb':
            self.mean = torch.FloatTensor([0.3337, 0.3064, 0.3171])
            self.std = torch.FloatTensor([0.2672, 0.2564, 0.2629])
        else: raise NotImplementedError()

    def detect(self):
        args = self.args
        
        if args.dataset == 'cifar10' or args.dataset == 'gtsrb':
            poison_set_dir = supervisor.get_poison_set_dir(args)
            
            clean_set_dir = os.path.join('clean_set', args.dataset, 'clean_split')
            clean_set_img_dir = os.path.join(clean_set_dir, 'data')
            clean_set_label_path = os.path.join(clean_set_dir, 'clean_labels')
            clean_set = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                                        label_path=clean_set_label_path, transforms=self.data_transform_aug)
            kwargs = {'num_workers': 4, 'pin_memory': True}
            clean_set_loader = torch.utils.data.DataLoader(
                clean_set,
                batch_size=self.batch_size, shuffle=True, **kwargs)
            
            test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
            test_set_img_dir = os.path.join(test_set_dir, 'data')
            test_set_label_path = os.path.join(test_set_dir, 'labels')
            test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                        label_path=test_set_label_path, transforms=self.data_transform)
            test_set_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=self.batch_size, shuffle=False, **kwargs)
        else: raise NotImplementedError()
        
        self.moth_core(self.args, self.model, clean_set_loader, test_set_loader, poison_set_dir)
    
    def moth_core(self, args, model, train_loader, test_loader, poison_set_dir):
        # assisting variables/parameters
        trigger_steps = 500
        warmup_steps  = 1
        cost   = 1e-3
        count  = np.zeros(2)
        WARMUP = True

        num_classes = get_num(args.dataset)
        img_rows, img_cols, img_channels = get_size(args.dataset)

        # matrices for recording distance changes
        mat_univ  = np.zeros((num_classes, num_classes)) # warmup distance
        mat_size  = np.zeros((num_classes, num_classes)) # trigger size
        mat_diff  = np.zeros((num_classes, num_classes)) # distance improvement
        mat_count = np.zeros((num_classes, num_classes)) # number of selected pairs

        mask_dict = {}
        pattern_dict = {}

        model.train()

        # set loss function and optimizer
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9,
                                    nesterov=True)

        # a subset for loss calculation during warmup
        for idx, (x_batch, y_batch) in enumerate(train_loader):
            if idx == 0:
                x_extra, y_extra = x_batch, y_batch
            else:
                x_extra = torch.cat((x_extra, x_batch))
                y_extra = torch.cat((y_extra, y_batch))
            if idx > 3:
                break

        num_samples = 10
        for i in range(num_classes):
            size = np.count_nonzero(y_extra == i)
            if size < num_samples:
                num_samples = size
        # assert (num_samples > 0)

        indices = []
        for i in range(num_classes):
            idx = np.where(y_extra == i)[0]
            indices.extend(list(idx[:num_samples]))
        x_extra = x_extra[indices]
        y_extra = y_extra[indices]
        assert (x_extra.size(0) == num_samples * num_classes)

        # set up trigger generation
        trigger = Trigger(
            model,
            args.dataset,
            steps=trigger_steps,
            asr_bound=0.99
        )
        trigger_combo = TriggerCombo(
            model,
            args.dataset,
            steps=trigger_steps
        )

        bound_size = img_rows * img_cols * img_channels / 4

        if self.type == 'adv':
            # attack parameters
            if args.dataset == 'cifar10':
                epsilon, k, a = 8 / 255, 7, 2 / 255
            elif args.dataset in ['svhn', 'gtsrb']:
                epsilon, k, a = 0.03, 8, 0.005
            elif args.dataset == 'lisa':
                epsilon, k, a = 0.1, 8, 0.02

        # hardening iterations
        max_warmup_steps = warmup_steps * num_classes
        steps_per_epoch = len(train_loader)
        max_steps = max_warmup_steps + self.epochs * steps_per_epoch

        step = 0
        source, target = 0, -1

        # start hardening
        print('=' * 80)
        print('start hardening...')
        time_start = time.time()
        for epoch in range(self.epochs):
            for (x_batch, y_batch) in train_loader:
                x_batch = x_batch.cuda()

                if self.type == 'nat':
                    x_adv = torch.clone(x_batch)
                elif self.type == 'adv':
                    x_adv = pgd_attack(
                        model,
                        deprocess(x_batch, args.dataset),
                        y_batch.cuda(),
                        self.mean,
                        self.std,
                        eps=epsilon,
                        alpha=a,
                        iters=k
                    )
                    x_adv = preprocess(x_adv, args.dataset)

                # update variables after warmup stage
                if step >= max_warmup_steps:
                    if WARMUP:
                        mat_diff /= np.max(mat_diff)
                    WARMUP = False
                    warmup_steps = 3

                # periodically update corresponding variables in each stage
                if (WARMUP and step % warmup_steps == 0) or \
                        (not WARMUP and (step - max_warmup_steps) % warmup_steps == 0):
                    if WARMUP:
                        target += 1
                        trigger_steps = 500
                    else:
                        if np.random.rand() < 0.3:
                            # randomly select a pair
                            source, target = np.random.choice(
                                np.arange(num_classes),
                                2,
                                replace=False
                            )
                        else:
                            # select a pair according to distance improvement
                            univ_sum = mat_univ + mat_univ.transpose()
                            diff_sum = mat_diff + mat_diff.transpose()
                            alpha = np.minimum(
                                0.1 * ((step - max_warmup_steps) / 100),
                                1
                            )
                            diff_sum = (1 - alpha) * univ_sum + alpha * diff_sum
                            source, target = np.unravel_index(np.argmax(diff_sum),
                                                            diff_sum.shape)

                            print('-' * 50)
                            print('fastest pair: {:d}-{:d}, improve: {:.2f}' \
                                .format(source, target, diff_sum[source, target]))

                        trigger_steps = 200

                    if source < target:
                        key = f'{source}-{target}'
                    else:
                        key = f'{target}-{source}'

                    print('-' * 50)
                    print('selected pair:', key)

                    # count the selected pair
                    if not WARMUP:
                        mat_count[source, target] += 1
                        mat_count[target, source] += 1

                    # use existing previous mask and pattern
                    if key in mask_dict:
                        init_mask = mask_dict[key]
                        init_pattern = pattern_dict[key]
                    else:
                        init_mask = None
                        init_pattern = None

                    # reset values
                    cost = 1e-3
                    count[...] = 0
                    mask_size_list = []

                if WARMUP:
                    # get a few samples from each label
                    indices = np.where(y_extra != target)[0]

                    # trigger inversion set
                    x_set = x_extra[indices]
                    y_set = torch.full((x_set.shape[0],), target)

                    # generate universal trigger
                    mask, pattern, speed \
                        = trigger.generate(
                        (num_classes, target),
                        x_set,
                        y_set,
                        attack_size=len(indices),
                        steps=trigger_steps,
                        init_cost=cost,
                        init_m=init_mask,
                        init_p=init_pattern
                    )

                    trigger_size = [mask.abs().sum().detach().cpu().numpy()] * 2

                    if trigger_size[0] < bound_size:
                        # choose non-target samples to stamp the generated trigger
                        indices = np.where(y_batch != target)[0]
                        length = int(len(indices) * self.warm_ratio)
                        choice = np.random.choice(indices, length, replace=False)

                        # stamp trigger
                        x_batch_adv = (1 - mask) \
                                    * deprocess(x_batch[choice], args.dataset) \
                                    + mask * pattern
                        x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)

                        x_adv[choice] = preprocess(x_batch_adv, args.dataset)

                    mask = mask.detach().cpu().numpy()
                    pattern = pattern.detach().cpu().numpy()

                    # record approximated distance improvement during warmup
                    for i in range(num_classes):
                        # mean loss change of samples of each source label
                        if i < target:
                            diff = np.mean(speed[i * num_samples: (i + 1) * num_samples])
                        elif i > target:
                            diff = np.mean(speed[(i - 1) * num_samples: i * num_samples])

                        if i != target:
                            mat_univ[i, target] = diff

                            # save generated triggers of a pair
                            src, tgt = i, target
                            key = f'{src}-{tgt}' if src < tgt else f'{tgt}-{src}'
                            if key not in mask_dict:
                                mask_dict[key] = mask[:1, ...]
                                pattern_dict[key] = pattern
                            else:
                                if src < tgt:
                                    mask_dict[key] = np.stack(
                                        [mask[:1, ...],
                                        mask_dict[key]],
                                        axis=0
                                    )
                                    pattern_dict[key] = np.stack(
                                        [pattern,
                                        pattern_dict[key]],
                                        axis=0
                                    )
                                else:
                                    mask_dict[key] = np.stack(
                                        [mask_dict[key],
                                        mask[:1, ...]],
                                        axis=0
                                    )
                                    pattern_dict[key] = np.stack(
                                        [pattern_dict[key],
                                        pattern],
                                        axis=0
                                    )

                            # initialize distance matrix entries
                            mat_size[i, target] = trigger_size[0]
                            mat_diff[i, target] = mat_size[i, target]
                else:
                    # get samples from source and target labels
                    idx_source = np.where(y_batch == source)[0]
                    idx_target = np.where(y_batch == target)[0]

                    # use a portion of source/target samples
                    length = int(min(len(idx_source), len(idx_target)) \
                                * self.portion)
                    if length > 0:
                        # dynamically adjust parameters
                        if (step - max_warmup_steps) % warmup_steps > 0:
                            if count[0] > 0 or count[1] > 0:
                                trigger_steps = 200
                                cost = 1e-3
                                count[...] = 0
                            else:
                                trigger_steps = 50
                                cost = 1e-2

                        # construct generation set for both directions
                        # source samples with target labels
                        # target samples with source labels
                        x_set = torch.cat((x_batch[idx_source],
                                        x_batch[idx_target]))
                        y_target = torch.full((len(idx_source),), target)
                        y_source = torch.full((len(idx_target),), source)
                        y_set = torch.cat((y_target, y_source))

                        # indicator vector for source/target
                        m_set = torch.zeros(x_set.shape[0])
                        m_set[:len(idx_source)] = 1

                        # generate a pair of triggers
                        mask, pattern \
                            = trigger_combo.generate(
                            (source, target),
                            x_set,
                            y_set,
                            m_set,
                            attack_size=x_set.shape[0],
                            steps=trigger_steps,
                            init_cost=cost,
                            init_m=init_mask,
                            init_p=init_pattern
                        )

                        trigger_size = mask.abs().sum(axis=(1, 2, 3)).detach() \
                            .cpu().numpy()

                        # operate on two directions
                        for cb in range(2):
                            if trigger_size[cb] < bound_size:
                                # choose samples to stamp the generated trigger
                                indices = idx_source if cb == 0 else idx_target
                                choice = np.random.choice(indices, length,
                                                        replace=False)

                                # stamp trigger
                                x_batch_adv \
                                    = (1 - mask[cb]) \
                                    * deprocess(x_batch[choice], args.dataset) \
                                    + mask[cb] * pattern[cb]
                                x_batch_adv = torch.clip(x_batch_adv, 0.0, 1.0)

                                x_adv[choice] = preprocess(x_batch_adv, args.dataset)

                        # save generated triggers of a pair
                        mask = mask.detach().cpu().numpy()
                        pattern = pattern.detach().cpu().numpy()
                        for cb in range(2):
                            if init_mask is None:
                                init_mask = mask[:, :1, ...]
                                init_pattern = pattern

                                if key not in mask_dict:
                                    mask_dict[key] = init_mask
                                    pattern_dict[key] = init_pattern
                            else:
                                if np.sum(mask[cb]) > 0:
                                    init_mask[cb] = mask[cb, :1, ...]
                                    init_pattern[cb] = pattern[cb]
                                    # save large trigger
                                    if np.sum(init_mask[cb]) \
                                            > np.sum(mask_dict[key][cb]):
                                        mask_dict[key][cb] = init_mask[cb]
                                        pattern_dict[key][cb] = init_pattern[cb]
                                else:
                                    # record failed generation
                                    count[cb] += 1

                        mask_size_list.append(
                            list(np.sum(3 * np.abs(init_mask), axis=(1, 2, 3)))
                        )

                    # periodically update distance related matrices
                    if (step - max_warmup_steps) % warmup_steps == warmup_steps - 1:
                        if len(mask_size_list) <= 0:
                            continue

                        # average trigger size of the current hardening period
                        mask_size_avg = np.mean(mask_size_list, axis=0)
                        if mat_size[source, target] == 0 \
                                or mat_size[target, source] == 0:
                            mat_size[source, target] = mask_size_avg[0]
                            mat_size[target, source] = mask_size_avg[1]
                            mat_diff = mat_size
                            mat_diff[mat_diff == -1] = 0
                        else:
                            # compute distance improvement
                            last_warm = mat_size[source, target]
                            mat_diff[source, target] \
                                += (mask_size_avg[0] - last_warm) / last_warm
                            mat_diff[source, target] /= 2

                            last_warm = mat_size[target, source]
                            mat_diff[target, source] \
                                += (mask_size_avg[1] - last_warm) / last_warm
                            mat_diff[target, source] /= 2

                            # update recorded trigger size
                            mat_size[source, target] = mask_size_avg[0]
                            mat_size[target, source] = mask_size_avg[1]

                x_batch = x_adv.detach()

                # train model
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch.cuda())
                loss.backward()
                optimizer.step()

                # evaluate and save model
                if (step + 1) % 10 == 0:
                    time_end = time.time()

                    total = 0
                    correct = 0
                    with torch.no_grad():
                        for (x_test, y_test) in test_loader:
                            x_test = x_test.cuda()
                            y_test = y_test.cuda()
                            total += y_test.size(0)

                            y_out = model(x_test)
                            _, y_pred = torch.max(y_out.data, 1)
                            correct += (y_pred == y_test).sum().item()
                    acc = correct / total

                    time_cost = time_end - time_start
                    print('*' * 120)
                    sys.stdout.write('step: {:4}/{:4} - {:.2f}s, ' \
                                    .format(step + 1, max_steps, time_cost) \
                                    + 'loss: {:.4f}, acc: {:.4f}\t'
                                    .format(loss, acc) \
                                    + 'trigger size: ({:.2f}, {:.2f})\n'
                                    .format(trigger_size[0], trigger_size[1]))
                    sys.stdout.flush()
                    print('*' * 120)

                    time_start = time.time()

                if step + 1 >= max_steps:
                    break

                step += 1

            if step + 1 >= max_steps:
                break
            
            tools.test(model, test_loader, poison_test=True, num_classes=self.num_classes, poison_transform=self.poison_transform)

        save_path = supervisor.get_model_dir(args, defense=True)
        torch.save(model.state_dict(), supervisor.get_model_dir(args, defense=True))
        print(f"Saved defended model to {save_path}")

        return model







