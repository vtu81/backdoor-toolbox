#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader, random_split, Dataset
from torchvision import transforms, datasets
import os
import config
from utils import supervisor
from utils.tools import test
from . import BackdoorDefense
from tqdm import tqdm
from .tools import generate_dataloader, AverageMeter, accuracy
import math

class NAD(BackdoorDefense):
    """
    Neural Attention Distillation

    Args:
        teacher_epochs (int): the number of finetuning epochs for a teacher model. Default: 10.
        erase_epochs (int): the number of epochs for erasing the poisoned student model via neural attention distillation. Default: 20.

    .. _Neural Attention Distillation:
        https://openreview.net/pdf?id=9l0K4OM-oXE


    .. _original source code:
        https://github.com/bboylyg/NAD

    """

    def __init__(self, args, teacher_epochs=10, erase_epochs=20):
        super().__init__(args)

        self.args = args
        self.teacher_epochs = teacher_epochs
        self.erase_epochs = erase_epochs
        
        self.p = 2 # power for AT
        self.batch_size = 64
        self.betas = [500, 1000, 1000] # hyperparams `betas` for AT loss (for ResNet and WideResNet archs)
        self.threshold_clean = 70.0 # don't save if clean acc drops too much

        self.folder_path = 'other_defenses_tool_box/results/NAD'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                                dataset_path=config.data_dir,
                                                batch_size=100,
                                                split='std_test',
                                                shuffle=False,
                                                drop_last=False,
                                                data_transform=self.data_transform)
        
        # 5% of the clean train set
        if args.dataset == 'cifar10':
            self.lr = 0.1
            self.ratio = 0.05 # ratio of training data to use
            full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'), train=True, download=True)
        elif args.dataset == 'gtsrb':
            self.lr = 0.02
            self.ratio = 0.2 # ratio of training data to use
            full_train_set = datasets.GTSRB(os.path.join(config.data_dir, 'gtsrb'), split='train', download=True)
        else: raise NotImplementedError()
        self.train_data = DatasetCL(self.ratio, full_dataset=full_train_set, transform=self.data_transform_aug)
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def detect(self):
        self.train_teacher()
        self.train_erase()

    def train_teacher(self):
        """
        Finetune the poisoned model with 5% of the clean train set to obtain a teacher model
        """
        # Load models
        print('----------- Network Initialization --------------')
        teacher = self.model
        teacher.train()
        print('finished teacher model init...')

        # initialize optimizer
        optimizer = torch.optim.SGD(teacher.module.parameters(),
                                    lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)

        # define loss functions
        criterion = nn.CrossEntropyLoss().cuda()

        print('----------- Train Initialization --------------')
        for epoch in range(0, self.teacher_epochs):

            self.adjust_learning_rate(optimizer, epoch, self.lr)

            if epoch == 0:
                # before training test firstly
                # self.test(teacher, criterion, epoch)
                test(teacher, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))

            self.train_step(self.train_loader, teacher, optimizer, criterion, epoch+1)

            # evaluate on testing set
            # acc_clean, acc_bad = self.test(teacher, criterion, epoch+1)
            # acc_clean, acc_bad = acc_clean[0], acc_bad[0]
            acc_clean, acc_bad = test(teacher, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))

            # remember best precision and save checkpoint
            # is_best = acc_clean[0] > self.threshold_clean
            # self.threshold_clean = min(acc_bad, self.threshold_clean)

            # best_clean_acc = acc_clean
            # best_bad_acc = acc_bad
            
            t_model_path = os.path.join(self.folder_path, 'NAD_T_%s.pt' % supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))
            self.save_checkpoint(teacher.module.state_dict(), True, t_model_path)

    def train_erase(self):
        """
        Erase the backdoor: teach the student (poisoned) model with the teacher model following NAD loss
        """
        # Load models
        print('----------- Network Initialization --------------')
        arch = supervisor.get_arch(self.args)
        
        teacher = arch(num_classes=self.num_classes)
        t_model_path = os.path.join(self.folder_path, 'NAD_T_%s.pt' % supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))
        checkpoint = torch.load(t_model_path)
        teacher.load_state_dict(checkpoint)
        teacher = nn.DataParallel(teacher)
        teacher = teacher.cuda()
        teacher.eval()

        student = arch(num_classes=self.num_classes)
        s_model_path = supervisor.get_model_dir(self.args)
        checkpoint = torch.load(s_model_path)
        student.load_state_dict(checkpoint)
        student = nn.DataParallel(student)
        student = student.cuda()
        student.train()
        print('finished student model init...')

        nets = {'snet': student, 'tnet': teacher}

        for param in teacher.module.parameters():
            param.requires_grad = False

        # initialize optimizer
        optimizer = torch.optim.SGD(student.module.parameters(),
                                    lr=self.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)

        # define loss functions
        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = AT(self.p)

        print('----------- Train Initialization --------------')
        for epoch in range(0, self.erase_epochs):

            self.adjust_learning_rate_erase(optimizer, epoch, self.lr)

            # train every epoch
            criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}

            if epoch == 0:
                # before training test firstly
                # self.test_erase(nets, criterions, self.betas, epoch)
                test(student, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))

            self.train_step_erase(self.train_loader, nets, optimizer, criterions, self.betas, epoch+1)

            # evaluate on testing set
            # acc_clean, acc_bad = self.test_erase(nets, criterions, self.betas, epoch+1)
            # acc_clean, acc_bad = acc_clean[0], acc_bad[0]
            acc_clean, acc_bad = test(student, test_loader=self.test_loader, poison_test=True, poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=self.source_classes, all_to_all=('all_to_all' in self.args.dataset))

            # remember best precision and save checkpoint
            is_best = acc_clean > self.threshold_clean
            self.threshold_clean = min(acc_bad, self.threshold_clean)

            best_clean_acc = acc_clean
            best_bad_acc = acc_bad
            
            erase_model_path = os.path.join(self.folder_path, 'NAD_E_%s.pt' % supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))
            self.save_checkpoint(student.module.state_dict(), is_best, erase_model_path)

    def test(self, model, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.eval()

        for idx, (img, target) in enumerate(self.test_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            with torch.no_grad():
                output = model(img)
                loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_clean = [top1.avg, top5.avg, losses.avg]

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (img, target) in enumerate(self.test_loader, start=1):
            img, target = img.cuda(), target.cuda()
            img = img[target != self.target_class]
            target = target[target != self.target_class]
            poison_img, poison_target = self.poison_transform.transform(img, target)

            with torch.no_grad():
                poison_output = model(poison_img)
                loss = criterion(poison_output, poison_target)

            prec1, prec5 = accuracy(poison_output, poison_target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_bd = [top1.avg, top5.avg, losses.avg]

        print('[Clean] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_clean[0], acc_clean[2]))
        print('[Bad] Prec@1: {:.2f}, Loss: {:.4f}'.format(acc_bd[0], acc_bd[2]))

        return acc_clean, acc_bd

    def test_erase(self, nets, criterions, betas, epoch):
        """
        Test the student model at erase step
        """
        top1 = AverageMeter()
        top5 = AverageMeter()

        snet = nets['snet']
        tnet = nets['tnet']

        criterionCls = criterions['criterionCls']
        criterionAT = criterions['criterionAT']

        snet.eval()

        for idx, (img, target) in enumerate(self.test_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            with torch.no_grad():
                output_s, _, _, _ = snet.forward(img, return_activation=True)

            prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        acc_clean = [top1.avg, top5.avg]

        cls_losses = AverageMeter()
        at_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for idx, (img, target) in enumerate(self.test_loader, start=1):
            img, target = img.cuda(), target.cuda()
            img = img[target != self.target_class]
            target = target[target != self.target_class]
            poison_img, poison_target = self.poison_transform.transform(img, target)

            with torch.no_grad():
                output_s, activation1_s, activation2_s, activation3_s = snet.forward(poison_img, return_activation=True)
                _, activation1_t, activation2_t, activation3_t = tnet.forward(poison_img, return_activation=True)

                at3_loss = criterionAT(activation3_s, activation3_t.detach()) * betas[2]
                at2_loss = criterionAT(activation2_s, activation2_t.detach()) * betas[1]
                at1_loss = criterionAT(activation1_s, activation1_t.detach()) * betas[0]
                at_loss = at3_loss + at2_loss + at1_loss
                cls_loss = criterionCls(output_s, poison_target)

            prec1, prec5 = accuracy(output_s, poison_target, topk=(1, 5))
            cls_losses.update(cls_loss.item(), poison_img.size(0))
            at_losses.update(at_loss.item(), poison_img.size(0))
            top1.update(prec1.item(), poison_img.size(0))
            top5.update(prec5.item(), poison_img.size(0))

        acc_bd = [top1.avg, top5.avg, cls_losses.avg, at_losses.avg]

        print('[clean]Prec@1: {:.2f}'.format(acc_clean[0]))
        print('[bad]Prec@1: {:.2f}'.format(acc_bd[0]))

        return acc_clean, acc_bd
    
    def train_step(self, train_loader, model, optimizer, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            output = model(img)

            loss = criterion(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch[{0}]: '
            'loss: {losses.avg:.4f}  '
            'prec@1: {top1.avg:.2f}  '
            'prec@5: {top5.avg:.2f}'.format(epoch, losses=losses, top1=top1, top5=top5))


    def train_step_erase(self, train_loader, nets, optimizer, criterions, betas, epoch):
        at_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        snet = nets['snet']
        tnet = nets['tnet']

        criterionCls = criterions['criterionCls']
        criterionAT = criterions['criterionAT']

        snet.train()

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            output_s, activation1_s, activation2_s, activation3_s = snet.forward(img, return_activation=True)
            _, activation1_t, activation2_t, activation3_t = tnet.forward(img, return_activation=True)

            cls_loss = criterionCls(output_s, target)
            at3_loss = criterionAT(activation3_s, activation3_t.detach()) * self.betas[2]
            at2_loss = criterionAT(activation2_s, activation2_t.detach()) * self.betas[1]
            at1_loss = criterionAT(activation1_s, activation1_t.detach()) * self.betas[0]
            at_loss = at1_loss + at2_loss + at3_loss + cls_loss

            prec1, prec5 = accuracy(output_s, target, topk=(1, 5))
            at_losses.update(at_loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            at_loss.backward()
            optimizer.step()

        print('Epoch[{0}]: '
            'AT_loss: {losses.avg:.4f}  '
            'prec@1: {top1.avg:.2f}  '
            'prec@5: {top5.avg:.2f}'.format(epoch, losses=at_losses, top1=top1, top5=top5))

    def adjust_learning_rate(self, optimizer, epoch, lr):
        # The learning rate is divided by 10 after every 2 epochs
        lr = lr * math.pow(10, -math.floor(epoch / 2))
        
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def adjust_learning_rate_erase(self, optimizer, epoch, lr):
        if epoch < 2:
            lr = lr
        elif epoch < 20:
            # lr = 0.01
            lr *= 0.1
        elif epoch < 30:
            # lr = 0.0001
            lr *= 0.001
        else:
            # lr = 0.0001
            lr *= 0.001
        print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_checkpoint(self, state, is_best, save_dir):
        if is_best:
            torch.save(state, save_dir)
            print('[info] save best model')

'''
AT with sum of absolute values with power p
code from: https://github.com/AberHu/Knowledge-Distillation-Zoo
'''
class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p):
		super(AT, self).__init__()
		self.p = p

	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am

class DatasetCL(Dataset):
    def __init__(self, ratio, full_dataset=None, transform=None):
        self.dataset = self.random_split(full_dataset=full_dataset, ratio=ratio)
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

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