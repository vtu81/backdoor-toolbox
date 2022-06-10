import numpy as np
import torch
import os
from torchvision import transforms,datasets
import argparse
import random
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from PIL import Image
from utils import supervisor, tools
import config


parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=config.parser_default['dataset'],
                    choices=config.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str,  required=False,
                    choices=config.parser_choices['poison_type'],
                    default=config.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=config.parser_choices['poison_rate'],
                    default=config.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=config.parser_choices['cover_rate'],
                    default=config.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False,
                    default=config.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float,  required=False, default=None)
parser.add_argument('-trigger', type=str, required=False, default=None)
parser.add_argument('-model_path', required=False, default=None)
parser.add_argument('-cleanser', type=str, required=False, default=None,
                    choices=['SCAn', 'AC', 'SS', 'Strip', 'CT', 'SPECTRE'])
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='1,2,3,4')
parser.add_argument('-seed', type=int, required=False, default=config.seed)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
if args.trigger is None:
    args.trigger = config.trigger_default[args.poison_type]

batch_size = 128
kwargs = {'num_workers': 4, 'pin_memory': True}


if args.dataset == 'cifar10':

    data_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

elif args.dataset == 'gtsrb':

    data_transform_aug = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

else:

    raise NotImplementedError('dataset %s not supported' % args.dataset)

if args.dataset == 'cifar10':
    num_classes = 10
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 200
    milestones = torch.tensor([100, 150])
    learning_rate = 0.1

elif args.dataset == 'cifar100':
    num_classes = 100
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)

elif args.dataset == 'gtsrb':
    num_classes = 43
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([40, 80])
    learning_rate = 0.1

elif args.dataset == 'imagenette':
    num_classes = 10
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)
else:
    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)


poison_set_dir = supervisor.get_poison_set_dir(args)
model_path = supervisor.get_model_dir(args, cleanse=(args.cleanser is not None))


arch = config.arch[args.dataset]
model = arch(num_classes=num_classes)
model.load_state_dict(torch.load(model_path))
model = nn.DataParallel(model)
model = model.cuda()
print("Evaluating model '{}'...".format(model_path))

# Set Up Test Set for Debug & Evaluation
test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
test_set_img_dir = os.path.join(test_set_dir, 'data')
test_set_label_path = os.path.join(test_set_dir, 'labels')
test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                             label_path=test_set_label_path, transforms=data_transform)
test_set_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size, shuffle=False, **kwargs)

# Poison Transform for Testing
poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                   target_class=config.target_class[args.dataset], trigger_transform=data_transform,
                                                   is_normalized_input=True,
                                                   alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                   trigger_name=args.trigger, args=args)

if args.poison_type == 'TaCT':
    source_classes = [config.source_class]
else:
    source_classes = None

tools.test(model=model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes)