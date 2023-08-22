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
import utils.ANP as models
from collections import OrderedDict


class ANP(BackdoorDefense):
    """
    Adversarial Neuron Pruning

    Args:
        

    .. _ANP:
        https://arxiv.org/abs/2110.14430


    .. _original source code:
        https://github.com/csdongxian/ANP_backdoor

    """
    
    def __init__(self, args, lr=0.2, anp_eps=0.4, anp_steps=1, anp_alpha=0.2, nb_iter=2000, print_every=500,
                 pruning_by='threshold', pruning_max=0.90, pruning_step=0.05, max_CA_drop=0.1):
        super().__init__(args)
        
        self.args = args
        self.lr = lr
        self.anp_eps = anp_eps
        self.anp_steps = anp_steps
        self.anp_alpha = anp_alpha
        self.nb_iter = nb_iter
        self.print_every = print_every # ??
        
        self.pruning_by = pruning_by
        self.pruning_max = pruning_max
        self.pruning_step = pruning_step
        
        self.max_CA_drop = max_CA_drop # maximum allowed clean accuracy drop during pruning
        

        self.folder_path = 'other_defenses_tool_box/results/ANP'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
            
        self.mask_scores_save_path = os.path.join(self.folder_path, '{}_mask_values.txt'.format(supervisor.get_dir_core(args)))
        
        self.valid_loader = generate_dataloader(dataset=self.dataset,
                                    dataset_path=config.data_dir,
                                    batch_size=100,
                                    split='valid',
                                    shuffle=True,
                                    drop_last=False)
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='test',
                                               shuffle=False)

    def detect(self):
        if not os.path.exists(self.mask_scores_save_path):
            self.optimize_mask()
        self.prune_neuron()
        return

    def optimize_mask(self):
        """
        Step 1: Generate prune mask.
        """
        args = self.args
        
        model_path = supervisor.get_model_dir(args)
        if 'resnet' in config.arch[args.dataset].__name__.lower():
            from utils.ANP import resnet_noisybn
            arch = resnet_noisybn.ResNet18
        else: raise NotImplementedError()
        net = arch(num_classes=self.num_classes)
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cuda')
            load_state_dict(net, orig_state_dict=state_dict)
            print("Loading from '{}'...".format(model_path))
        else:
            print("Model '{}' not found.".format(model_path))
        net = net.cuda()
        net.eval()
        
        parameters = list(net.named_parameters())
        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=self.lr, momentum=0.9)
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=self.anp_eps / self.anp_steps)
        criterion = torch.nn.CrossEntropyLoss().cuda()
        
        print('Iter \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        nb_repeat = int(np.ceil(self.nb_iter / self.print_every))
        for i in range(nb_repeat):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            train_loss, train_acc = self.mask_train(model=net, criterion=criterion, data_loader=self.test_loader,
                                            mask_opt=mask_optimizer, noise_opt=noise_optimizer)
            cl_test_loss, cl_test_acc = anp_test(model=net, criterion=criterion, data_loader=self.test_loader)
            po_test_loss, po_test_acc = anp_test(model=net, criterion=criterion, data_loader=self.test_loader, poison_transform=self.poison_transform)
            end = time.time()
            print('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
                (i + 1) * self.print_every, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
                cl_test_loss, cl_test_acc))
            
        
        save_mask_scores(net.state_dict(), self.mask_scores_save_path)
        print(f"Saved mask scores to '{self.mask_scores_save_path}'.")

    def prune_neuron(self):
        """
        Step 2: Prune.
        """
        args = self.args
        net = self.model.module
        criterion = torch.nn.CrossEntropyLoss().cuda()
        
        mask_values = read_data(self.mask_scores_save_path)
        mask_values = sorted(mask_values, key=lambda x: float(x[2]))
        # print('No. \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        # cl_loss, cl_acc = anp_test(model=net, criterion=criterion, data_loader=self.test_loader)
        # po_loss, po_acc = anp_test(model=net, criterion=criterion, data_loader=self.test_loader, poison_transform=self.poison_transform)
        # print('0 \t None     \t None \t\t None \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(po_loss, po_acc, cl_loss, cl_acc))
        print("[Original]")
        self.ori_CA, self.ori_ASR = test(self.model, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))
        
        print("\n[Prunning]")
        if self.pruning_by == 'threshold':
            results = self.evaluate_by_threshold(net, mask_values, criterion=criterion)
        else:
            results = self.evaluate_by_number(net, mask_values, criterion=criterion)
        file_name = os.path.join(self.folder_path, '{}_pruning_by_{}.txt'.format(supervisor.get_dir_core(args), self.pruning_by))
        with open(file_name, "w") as f:
            f.write('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC\n')
            f.writelines(results)
        
        save_path = supervisor.get_model_dir(self.args, defense=True)
        print(f"Saved to {save_path}")
        torch.save(self.model.module.state_dict(), save_path)
        
        return
    
    
    def mask_train(self, model, criterion, mask_opt, noise_opt, data_loader):
        model.train()
        total_correct = 0
        total_loss = 0.0
        nb_samples = 0
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            nb_samples += images.size(0)

            # step 1: calculate the adversarial perturbation for neurons
            if self.anp_eps > 0.0:
                reset(model, rand_init=True, anp_eps=self.anp_eps)
                for _ in range(self.anp_steps):
                    noise_opt.zero_grad()

                    include_noise(model)
                    output_noise = model(images)
                    loss_noise = - criterion(output_noise, labels)

                    loss_noise.backward()
                    sign_grad(model)
                    noise_opt.step()

            # step 2: calculate loss and update the mask values
            mask_opt.zero_grad()
            if self.anp_eps > 0.0:
                include_noise(model)
                output_noise = model(images)
                loss_rob = criterion(output_noise, labels)
            else:
                loss_rob = 0.0

            exclude_noise(model)
            output_clean = model(images)
            loss_nat = criterion(output_clean, labels)
            loss = self.anp_alpha * loss_nat + (1 - self.anp_alpha) * loss_rob

            pred = output_clean.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
            loss.backward()
            mask_opt.step()
            clip_mask(model)

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / nb_samples
        return loss, acc

    def evaluate_by_number(self, model, mask_values, criterion):
        results = []
        nb_max = int(np.ceil(self.pruning_max))
        nb_step = int(np.ceil(self.pruning_step))
        for start in range(0, nb_max + 1, nb_step):
            i = start
            for i in range(start, start + nb_step):
                pruning(model, mask_values[i])
            layer_name, neuron_idx, value = mask_values[i][0], mask_values[i][1], mask_values[i][2]
            # cl_loss, cl_acc = anp_test(model=model, criterion=criterion, data_loader=self.test_loader)
            # po_loss, po_acc = anp_test(model=model, criterion=criterion, data_loader=self.test_loader, poison_transform=self.poison_transform)
            # print('{} \t {} \t {} \t\t {:.3f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            #     i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
            # results.append('{} \t {} \t {} \t\t {:.3f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            #     i+1, layer_name, neuron_idx, value, po_loss, po_acc, cl_loss, cl_acc))
            
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
            # layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
            # cl_loss, cl_acc = anp_test(model=model, criterion=criterion, data_loader=self.test_loader)
            # po_loss, po_acc = anp_test(model=model, criterion=criterion, data_loader=self.test_loader, poison_transform=self.poison_transform)
            # print('{:d} \t {} \t {} \t\t {:.3f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'.format(
            #     start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
            # results.append('{:d} \t {} \t {} \t\t {:.3f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}\n'.format(
            #     start, layer_name, neuron_idx, threshold, po_loss, po_acc, cl_loss, cl_acc))
            
            print('pruned neurons num = {:d}, threshold = {:.3f}'.format(start, threshold))
            CA, ASR = test(model, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))
            results.append('pruned neurons num = {:d}, threshold = {:.3f} \t CA = {:.2f} \t ASR = {:.2f}\n'.format(start, threshold, CA * 100, ASR * 100))
            if self.ori_CA - CA > self.max_CA_drop: break
        
        return results




"""
Utility functions copied from `optimize_mask_cifar.py`.
"""
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
    params = [param for name, param in model.named_parameters() if 'neuron_mask' in name]
    with torch.no_grad():
        for param in params:
            param.clamp_(lower, upper)


def sign_grad(model):
    noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
    for p in noise:
        p.grad.data = torch.sign(p.grad.data)


def perturb(model, is_perturbed=True):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.perturb(is_perturbed=is_perturbed)


def include_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.include_noise()


def exclude_noise(model):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.exclude_noise()


def reset(model, rand_init, anp_eps):
    for name, module in model.named_modules():
        if isinstance(module, models.NoisyBatchNorm2d) or isinstance(module, models.NoisyBatchNorm1d):
            module.reset(rand_init=rand_init, eps=anp_eps)


def anp_test(model, criterion, data_loader, poison_transform=None):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.cuda(), labels.cuda()
            if poison_transform is not None:
                images, labels = poison_transform.transform(images, labels)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


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
        

"""
Utility functions copied from `prune_neuron_cifar.py`.
"""
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



