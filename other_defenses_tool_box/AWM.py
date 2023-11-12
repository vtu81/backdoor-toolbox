#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import time
import config
from utils import supervisor
from utils.tools import test
from . import BackdoorDefense
from .tools import to_list, generate_dataloader, val_atk
import numpy as np
import pandas as pd
from collections import OrderedDict
import utils.AWM.models as models
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset


class AWM(BackdoorDefense):
    """
    Adversarial Weight Masking

    Args:
        

    .. AWM:
        https://openreview.net/forum?id=Yb3dRKY170h


    .. _original source code:
        https://github.com/jinghuichen/AWM

    """
    
    def __init__(self, args, lr1, lr2, outer, inner, shrink_steps, batch_size=128, trigger_norm=1000, alpha=0.9, gamma=1e-8, lr_decay=False):
        super().__init__(args)
        
        self.args = args
        self.lr1 = lr1
        self.lr2 = lr2
        self.inner = inner
        self.outer = outer
        self.shrink_steps = shrink_steps
        self.lr_decay = lr_decay
        self.batch_size = batch_size
        self.trigger_norm = trigger_norm
        self.alpha = alpha
        self.gamma = gamma
        
        
        

        self.folder_path = 'other_defenses_tool_box/results/AWM'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
            
        
        self.valid_loader = generate_dataloader(dataset=self.dataset,
                                    dataset_path=config.data_dir,
                                    batch_size=100,
                                    split='valid',
                                    shuffle=True,
                                    drop_last=False)
        self.valid_dataset = self.valid_loader.dataset
        self.inner_iters = int(len(self.valid_dataset) / self.batch_size) * self.inner
        random_sampler = RandomSampler(data_source=self.valid_dataset, replacement=True,
                                   num_samples=self.inner_iters * self.batch_size)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, sampler=random_sampler, num_workers=4)
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=self.batch_size,
                                               split='test',
                                               shuffle=False)

    def detect(self):
        
        # Load backdoor model
        model_path = supervisor.get_model_dir(self.args)
        if 'resnet' in config.arch[self.args.dataset].__name__.lower():
            from utils.AWM.models import resnet18
            arch = resnet18
        else: raise NotImplementedError()
        net = arch(num_classes=self.num_classes, norm_layer=nn.BatchNorm2d)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cuda')
            load_state_dict(net, orig_state_dict=state_dict)
            print("Loading from '{}'...".format(model_path))
        else:
            print("Model '{}' not found.".format(model_path))
        net = net.cuda()
        net.eval()
        
        
        criterion = torch.nn.CrossEntropyLoss().cuda()
        
        from utils.AWM.models import MaskedConv2d
        for name, module in net.named_modules():
            if isinstance(module, MaskedConv2d):
                module.include_mask()
            
        parameters = list(net.named_parameters())
        mask_params = [v for n, v in parameters if "mask" in n]
        mask_names = [n for n, v in parameters if "mask" in n]
        mask_optimizer = torch.optim.Adam(mask_params, lr = self.lr1)
        
        
        
        # Optional to use shrink_steps to reduce the size of the model
        for i in range(self.shrink_steps):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            train_loss, train_acc = shrink(model=net, criterion=criterion, data_loader=self.valid_loader, mask_opt=mask_optimizer, gamma=self.gamma)
            # cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
            # po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
            end = time.time()
            
        
        mask_optimizer = torch.optim.Adam(mask_params, lr = self.lr2)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=mask_optimizer, gamma=0.9)
        
        # AWM
        for i in range(self.outer):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            
            train_loss, train_acc = mask_train(model=net, criterion=criterion, data_loader=self.valid_loader, mask_opt=mask_optimizer, trigger_norm=self.trigger_norm, alpha=self.alpha, gamma=self.gamma)
            # cl_test_loss, cl_test_acc = test(model=net, criterion=criterion, data_loader=clean_test_loader)
            # po_test_loss, po_test_acc = test(model=net, criterion=criterion, data_loader=poison_test_loader)
            end = time.time()

            print(f"Outer Iteration: {i + 1}/{self.outer}", " | lr: ", lr, " | train_loss: ", train_loss, " | train_acc: ", train_acc)
            CA, ASR = test(net, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))
            # print('Iter \t\t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
            # print('EPOCHS {} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            #     (i + 1) * self.inner_iters, po_test_loss, po_test_acc,
            #     cl_test_loss, cl_test_acc))
            
            if self.lr_decay:
                my_lr_scheduler.step()
        return



def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']
    if "state_dict" in orig_state_dict.keys():
        orig_state_dict = orig_state_dict["state_dict"]

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        elif 'running_mean_noisy' in k or 'running_var_noisy' in k or 'num_batches_tracked_noisy' in k:
            new_state_dict[k] = orig_state_dict[k[:-6]].clone().detach()
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)
    
def clip_mask(model, lower=0.0, upper=1.0):
    params = [param for name, param in model.named_parameters() if 'mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.MaskedConv2d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.MaskedConv2d):
            module.exclude_noise()


def reset(model, rand_init):
    for name, module in model.named_modules():
        if isinstance(module, models.MaskedConv2d):
            module.reset(rand_init=rand_init, eps=0.4)

def shrink(model, criterion, mask_opt, data_loader, gamma):
    model.eval()
    nb_samples = 0

    for i, (images, labels) in enumerate(data_loader):

        images, labels = images.cuda(), labels.cuda()
        nb_samples += images.size(0)

        output_clean = model(images)

        loss_nat = criterion(output_clean, labels)
        L1, L2 = Regularization(model)

        loss = gamma * L1 + loss_nat

        mask_opt.zero_grad()
        loss.backward()
        mask_opt.step()
        clip_mask(model)
    return 0, 0

def mask_train(model, criterion, mask_opt, data_loader, trigger_norm, alpha, gamma):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0

    batch_pert = torch.zeros([1,3,32,32], requires_grad=True, device='cuda')

    batch_opt = torch.optim.SGD(params=[batch_pert], lr=10)

    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.cuda(), labels.cuda()

        # step 1: calculate the adversarial perturbation for images
        ori_lab = torch.argmax(model.forward(images),axis = 1).long()
        per_logits = model.forward(images + batch_pert)
        loss = F.cross_entropy(per_logits, ori_lab, reduction='mean')
        loss_regu = torch.mean(-loss)

        batch_opt.zero_grad()
        loss_regu.backward(retain_graph = True)
        batch_opt.step()

    pert = batch_pert * min(1, trigger_norm / torch.sum(torch.abs(batch_pert)))
    pert = pert.detach()

    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.cuda(), labels.cuda()
        nb_samples += images.size(0)
        
        perturbed_images = torch.clamp(images + pert[0], min=0, max=1)
        
        # step 2: calculate noisy loss and clean loss
        mask_opt.zero_grad()
        
        output_noise = model(perturbed_images)
        
        output_clean = model(images)
        pred = torch.argmax(output_clean, axis = 1).long()

        loss_rob = criterion(output_noise, labels)
        loss_nat = criterion(output_clean, labels)
        L1, L2 = Regularization(model)

        # print("loss_noise | ", loss_rob.item(), " | loss_clean | ", loss_nat.item(), " | L1 | ", L1.item())
        loss = alpha * loss_nat + (1 - alpha) * loss_rob + gamma * L1

        pred = output_clean.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()

        mask_opt.step()
        clip_mask(model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

def Regularization(model):
    L1=0
    L2=0
    for name, param in model.named_parameters():
        if 'mask' in name:
            L1 += torch.sum(torch.abs(param))
            L2 += torch.norm(param, 2)
    return L1, L2