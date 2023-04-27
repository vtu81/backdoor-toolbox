import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os 
from torch.autograd import Variable
import torchvision.datasets as datasets
from .none import compromised_detection, reset_neurons
from . import BackdoorDefense
from utils import supervisor, tools
import config


# python -u run_none.py --dataset cifar10 --arch resnet18 --poison_type single_target --none_lr 1e-4 --max_reset_fraction 0.03 --poison_rate 0.05 --epoch_num_1 200 --epoch_num_2 20
class NONE(BackdoorDefense):
    name: str = 'none'

    def __init__(self, args, epoch_num_1=320, epoch_num_2=20, round_num=1,
                 lr_decay_every=80, base_lr=0.1, none_lr=0.1, max_reset_fraction=0.05,
                 lamda_l=0.1, lamda_h=0.9, num_for_detect_biased=-1, batch_size=128):
        super().__init__(args)
        
        self.args = args
        self.epoch_num_1 = epoch_num_1
        self.epoch_num_2 = epoch_num_2
        self.round_num = round_num
        self.lr_decay_every = lr_decay_every
        self.base_lr = base_lr
        self.none_lr = none_lr
        self.max_reset_fraction = max_reset_fraction
        self.lamda_l = lamda_l
        self.lamda_h = lamda_h
        self.num_for_detect_biased = num_for_detect_biased
        
        if args.dataset == 'imagenet':
            kwargs = {'num_workers': 32, 'pin_memory': True}
        else:
            kwargs = {'num_workers': 4, 'pin_memory': True}
        
        if args.dataset == 'cifar10' or args.dataset == 'gtsrb':
            # Set Up Poisoned Set
            self.poison_set_dir = supervisor.get_poison_set_dir(args)
            if os.path.exists(os.path.join(self.poison_set_dir, 'data')): # if old version
                poisoned_set_img_dir = os.path.join(self.poison_set_dir, 'data')
            if os.path.exists(os.path.join(self.poison_set_dir, 'imgs')): # if new version
                poisoned_set_img_dir = os.path.join(self.poison_set_dir, 'imgs')
            poisoned_set_label_path = os.path.join(self.poison_set_dir, 'labels')
            poison_indices_path = os.path.join(self.poison_set_dir, 'poison_indices')


            print('dataset : %s' % poisoned_set_img_dir)

            self.poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                            label_path=poisoned_set_label_path, transforms=self.data_transform_aug)

            self.poisoned_set_loader = torch.utils.data.DataLoader(
                self.poisoned_set,
                batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)
            
            self.poisoned_set_no_shuffled = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                            label_path=poisoned_set_label_path, transforms=self.data_transform)
            
            self.poisoned_set_loader_no_shuffled = torch.utils.data.DataLoader(
                self.poisoned_set_no_shuffled,
                batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)
            
            
            # Set Up Test Set for Debug & Evaluation
            test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
            test_set_img_dir = os.path.join(test_set_dir, 'data')
            test_set_label_path = os.path.join(test_set_dir, 'labels')
            self.test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                        label_path=test_set_label_path, transforms=self.data_transform)
            self.test_set_loader = torch.utils.data.DataLoader(
                self.test_set,
                batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

            # Poison Transform for Testing
            self.poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                            target_class=config.target_class[args.dataset], trigger_transform=self.data_transform,
                                                            is_normalized_input=True,
                                                            alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                            trigger_name=args.trigger, args=args)

        else: raise NotImplementedError()
        
    def detect(self):
        args = self.args
        arch = supervisor.get_arch(args).__name__.lower()
        
        if arch == 'resnet18':
            optimizer = optim.SGD(self.model.parameters(), lr=self.base_lr, momentum=0.9, weight_decay=1e-4)
        else: raise NotImplementedError()
        criterion = nn.CrossEntropyLoss()

        
        loader = self.poisoned_set_loader
        
        base_path = os.path.join(self.poison_set_dir, f"base_{supervisor.get_model_name(args, defense=True)}")
        if os.path.exists(base_path):
            self.model.module.load_state_dict(torch.load(base_path))
            print(f"Loaded base model from {base_path}.")
        else:
            for epoch in range(1, self.epoch_num_1+1):
                adjust_learning_rate(optimizer, self.lr_decay_every, epoch)
                train_epoch(self.model, optimizer, epoch, loader, criterion)
            torch.save(self.model.module.state_dict(), base_path)
            print(f"Saved base model to {base_path}.")

        tools.test(model=self.model, test_loader=self.test_set_loader, poison_test=True if args.poison_type != 'none' else False,
                        poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=[config.source_class] if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent' else None, all_to_all='all_to_all' in args.poison_type)



        for param_group in optimizer.param_groups:
            param_group['lr'] = self.none_lr

        for round_id in range(1, self.round_num+1):
            
            selected_neuros, poison_sample_index = compromised_detection.analyze_neuros(self.model.module, arch, self.max_reset_fraction,
                                                                self.lamda_l, self.lamda_h, self.num_classes,
                                                                self.num_for_detect_biased,
                                                                self.poisoned_set_loader_no_shuffled)
            
            indices = list(set([a for a in range(len(self.poisoned_set))]) - set(poison_sample_index))
            loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.poisoned_set, indices), batch_size=128, shuffle=True)
            
            self.model = reset_neurons.reset(self.model, arch, selected_neuros)

            for epoch in range(1, self.epoch_num_2+1):
                adjust_learning_rate(optimizer, self.lr_decay_every, epoch)
                train_epoch(self.model, optimizer, epoch, loader, criterion, freeze_bn=True)
                
                tools.test(model=self.model, test_loader=self.test_set_loader, poison_test=True if args.poison_type != 'none' else False,
                        poison_transform=self.poison_transform, num_classes=self.num_classes, source_classes=[config.source_class] if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent' else None, all_to_all='all_to_all' in args.poison_type)
            
            save_path = supervisor.get_model_dir(args, defense=True)
            torch.save(self.model.module.state_dict(), save_path)
            print(f"Saved defended model to {save_path}.")


def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def adjust_learning_rate(optimizer, lr_decay_every, epoch):
    if epoch%lr_decay_every==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

def train_epoch(model, optimizer, epoch, loader, criterion, freeze_bn=False):
    model.train()
    if freeze_bn:
        model.apply(fix_bn)
    
    for batch_idx, (data, target) in enumerate(loader):

        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # if batch_idx % 100 == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
        #         epoch, batch_idx * len(data), len(loader.dataset),
        #         100. * batch_idx / len(loader), loss.item(),
        #         optimizer.param_groups[0]['lr']))
        
    print('Train Epoch: {}\tLoss: {:.6f}\tLR: {}'.format(
            epoch, loss.item(), optimizer.param_groups[0]['lr']))

    # if (epoch+1)%20 == 0:
        # torch.save(model.state_dict(), SAVE_PATH.split("latest.")[0] + "epoch" + str(epoch+1) +'.pth')