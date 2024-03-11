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
import utils.RNP.models as models
from collections import OrderedDict
from torch.utils.data import Subset, DataLoader


class RNP(BackdoorDefense):
    """
    Reconstructive Neuron Pruning

    Args:
        

    .. _RNP:
        https://arxiv.org/pdf/2305.14876.pdf


    .. _original source code:
        https://github.com/bboylyg/RNP

    """
    
    def __init__(self, args, schedule=[10, 20], batch_size=128, momentum=0.9, weight_decay=5e-4, alpha=0.2, clean_threshold=0.20, unlearning_lr=0.01, recovering_lr=0.2, unlearning_epochs=20, recovering_epochs=20, pruning_by='threshold', pruning_max=0.90, pruning_step=0.05, max_CA_drop=0.1):
        super().__init__(args)
        
        self.args = args
        self.schedule = schedule
        self.batch_size = batch_size
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.clean_threshold = clean_threshold

        self.unlearning_lr = unlearning_lr
        self.recovering_lr = recovering_lr
        self.unlearning_epochs = unlearning_epochs
        self.recovering_epochs = recovering_epochs
        self.pruning_by = pruning_by
        self.pruning_max = pruning_max
        self.pruning_step = pruning_step

        self.max_CA_drop = max_CA_drop

        self.folder_path = 'other_defenses_tool_box/results/RNP'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
            
        self.mask_scores_save_path = os.path.join(self.folder_path, '{}_mask_values.txt'.format(supervisor.get_dir_core(args)))
        
        self.valid_loader = generate_dataloader(dataset=self.dataset,
                                    dataset_path=config.data_dir,
                                    batch_size=batch_size,
                                    split='valid',
                                    shuffle=True,
                                    drop_last=False)

        # Randomly select 10% of the validation set as the new validation set
        # random_sampler = torch.utils.data.RandomSampler(self.valid_loader.dataset, num_samples=int(len(self.valid_loader.dataset) * 0.25), replacement=False)
        # self.valid_loader = DataLoader(self.valid_loader.dataset, batch_size=self.batch_size, shuffle=False, sampler=random_sampler, num_workers=4)
        
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='test',
                                               shuffle=False)

    def detect(self):
        # Load backdoor model
        model_path = supervisor.get_model_dir(self.args)
        if 'resnet' in config.arch[self.args.dataset].__name__.lower():
            from utils.RNP.models import resnet18
            arch = resnet18
        else: raise NotImplementedError()
        net = arch(num_classes=self.num_classes, norm_layer=None)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cuda')
            load_state_dict(net, orig_state_dict=state_dict)
            print("Loading from '{}'...".format(model_path))
        else:
            print("Model '{}' not found.".format(model_path))
        net = net.cuda()
        
        
        # Step 1: Unlearning
        criterion = torch.nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.unlearning_lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.schedule, gamma=0.1)
        
        print('----------- Model Unlearning --------------')
        for epoch in range(0, self.unlearning_epochs + 1):
            start = time.time()
            lr = optimizer.param_groups[0]['lr']
            train_loss, train_acc = train_step_unlearning(model=net, criterion=criterion, optimizer=optimizer, data_loader=self.valid_loader)
            scheduler.step()
            end = time.time()
            
            print(f"Epoch: {epoch}/{self.unlearning_epochs}, Train Acc: {train_acc}, Time: {end - start}s")
            CA, ASR = test(net, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))
            

            # if train_acc <= self.clean_threshold: # train acc or test acc? not sure
            if CA <= self.clean_threshold:
                unlearned_save_path = f"{supervisor.get_poison_set_dir(self.args)}/defended_RNP_unlearned_{supervisor.get_model_name(self.args)}"
                print(f"Saved to {unlearned_save_path}")
                torch.save(net.state_dict(), unlearned_save_path)
                break
            
        
        
        # Step 2: Recover from unlearned model checkpoint
        unlearned_save_path = f"{supervisor.get_poison_set_dir(self.args)}/defended_RNP_unlearned_{supervisor.get_model_name(self.args)}"
        checkpoint = torch.load(unlearned_save_path, map_location='cuda')
        print(f"Loaded from {unlearned_save_path}.")

        if 'resnet' in config.arch[self.args.dataset].__name__.lower():
            from utils.RNP.models import resnet18
            arch = resnet18
        else: raise NotImplementedError()
        unlearned_model = arch(num_classes=10, norm_layer=models.MaskBatchNorm2d)
        load_state_dict(unlearned_model, orig_state_dict=checkpoint)
        unlearned_model = unlearned_model.cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()

        parameters = list(unlearned_model.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=self.recovering_lr, momentum=0.9)

        # Recovering
        print('----------- Model Recovering --------------')
        for epoch in range(1, self.recovering_epochs + 1):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            train_loss, train_acc = train_step_recovering(unlearned_model=unlearned_model, criterion=criterion, data_loader=self.valid_loader, mask_opt=mask_optimizer, alpha=self.alpha)
            end = time.time()
            
            print(f"Epoch: {epoch}/{self.recovering_epochs}, Train Acc: {train_acc}, Time: {end - start}s")
            CA, ASR = test(unlearned_model, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))
        
        save_mask_scores(unlearned_model.state_dict(), self.mask_scores_save_path)

        # del unlearned_model, net
        
        
        
        # Step 3: pruning
        print('----------- Backdoored Model Pruning --------------')
        # Load backdoor model
        model_path = supervisor.get_model_dir(self.args)
        if 'resnet' in config.arch[self.args.dataset].__name__.lower():
            from utils.RNP.models import resnet18
            arch = resnet18
        else: raise NotImplementedError()
        net = arch(num_classes=self.num_classes, norm_layer=None)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cuda')
            load_state_dict(net, orig_state_dict=state_dict)
            print("Loading from '{}'...".format(model_path))
        else:
            print("Model '{}' not found.".format(model_path))
        net = net.cuda()

        criterion = torch.nn.CrossEntropyLoss().cuda()

        mask_values = read_data(self.mask_scores_save_path)
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))

        print("[Original]")
        self.ori_CA, self.ori_ASR = test(net, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))

        if self.pruning_by == 'threshold':
            results = self.evaluate_by_threshold(
                net, mask_values, criterion=criterion
            )
        else:
            results = self.evaluate_by_number(
                net, mask_values, criterion=criterion
            )
            
        print('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
        print(results)
        
        file_name = os.path.join(self.folder_path, '{}_pruning_by_{}.txt'.format(supervisor.get_dir_core(self.args), self.pruning_by))
        with open(file_name, "w") as f:
            f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
            f.writelines(results)
            print(f"Saved to {file_name}.")
        
        return

    def evaluate_by_number(self, model, mask_values, criterion):
        results = []
        nb_max = int(np.ceil(self.pruning_max * len(mask_values)))
        nb_step = int(np.ceil(self.pruning_step * len(mask_values)))
        for start in range(0, nb_max + 1, nb_step):
            i = start
            for i in range(start, start + nb_step):
                pruning(model, mask_values[i])
            layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
            
            print('pruned neurons num = {:d}, threshold = {:.3f}'.format(start, value))
            CA, ASR = test(model, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))
            results.append('pruned neurons = {:d}, threshold = {:.3f} \t CA = {:.2f} \t ASR = {:.2f}\n'.format(start, value, CA * 100, ASR * 100))
            if self.ori_CA - CA > self.max_CA_drop: break
            
        return results


    def evaluate_by_threshold(self, model, mask_values, criterion):
        results = []
        thresholds = np.arange(0, self.pruning_max + self.pruning_step, self.pruning_step)
        start = 0
        for threshold in thresholds:
            idx = start
            for idx in range(start, len(mask_values)):
                if float(mask_values[idx][2]) <= threshold:
                    pruning(model, mask_values[idx])
                    start += 1
                else:
                    break

            print('pruned neurons num = {:d}, threshold = {:.3f}'.format(start, threshold))
            CA, ASR = test(model, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))
            results.append('pruned neurons num = {:d}, threshold = {:.3f} \t CA = {:.2f} \t ASR = {:.2f}\n'.format(start, threshold, CA * 100, ASR * 100))
            if self.ori_CA - CA > self.max_CA_drop: break
        
        return results




"""
Utility functions copied from `optimize_mask_cifar.py`.
"""
def train_step_unlearning(model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        (-loss).backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def train_step_recovering(unlearned_model, criterion, mask_opt, data_loader, alpha):
    unlearned_model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.cuda(), labels.cuda()
        nb_samples += images.size(0)

        mask_opt.zero_grad()
        output = unlearned_model(images)
        loss = criterion(output, labels)
        loss = alpha * loss

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        mask_opt.step()
        clip_mask(unlearned_model)

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc

def load_state_dict(net, orig_state_dict):
    if 'state_dict' in orig_state_dict.keys():
        orig_state_dict = orig_state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in net.state_dict().items():
        if k in orig_state_dict.keys():
            new_state_dict[k] = orig_state_dict[k]
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)


def clip_mask(unlearned_model, lower=0.0, upper=1.0):
    params = [param for name, param in unlearned_model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def save_mask_scores(state_dict, file_name):
    mask_values = []
    count = 0
    for name, param in state_dict.items():
        if 'neuron_mask' in name:
            for idx in range(param.size(0)):
                neuron_name = '.'.join(name.split('.')[:-1])
                mask_values.append('{} \t {} \t {} \t {:.4f} \n'.format(count, neuron_name, idx, param[idx].item()))
                count += 1
    with open(file_name, "w") as f:
        f.write('No \t Layer Name \t Neuron Idx \t Mask Score \n')
        f.writelines(mask_values)

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    mask_values = list(zip(layer, idx, value))
    return mask_values

def pruning(net, neuron):
    state_dict = net.state_dict()
    weight_name = '{}.{}'.format(neuron[0], 'weight')
    state_dict[weight_name][int(neuron[1])] = 0.0
    net.load_state_dict(state_dict)