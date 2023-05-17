import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import argparse
import random
import config
from utils import default_args, tools

"""
<Datasets>
GTSRB, CIFAR10, Imagenette, Imagenet, Ember
"""

parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, required=False, default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-clean_budget', type=int, default=2000)
# by defaut :  we assume 2000 clean samples for defensive purpose

args = parser.parse_args()

tools.setup_seed(0)

"""
Get Data Set
"""
data_dir = './data'  # directory to save standard clean set

if args.dataset == 'gtsrb':
    data_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])
    clean_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split='test',
                               transform=data_transform, download=True)
    img_size = 32
    num_classes = 43
elif args.dataset == 'cifar10':
    data_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    clean_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=False,
                                 download=True, transform=data_transform)
    img_size = 32
    num_classes = 10
else:
    print('<Undefined> Dataset = %s' % args.dataset)
    exit(0)

"""
Generate Clean Split
"""

root_dir = 'clean_set'
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

root_dir = os.path.join(root_dir, args.dataset)
if not os.path.exists(root_dir):
    os.mkdir(root_dir)

test_split_dir = os.path.join(root_dir, 'noisy_test_split')  # test samples for evaluation & debug purpose
if not os.path.exists(test_split_dir):
    os.mkdir(test_split_dir)

test_split_img_dir = os.path.join(test_split_dir, 'data')  # to save img
if not os.path.exists(test_split_img_dir):
    os.mkdir(test_split_img_dir)


def AddNoise(img_tensor, noise_magnitude=0.05):
    # generate the noise tensor and add it to the image tensor
    noise = torch.randn_like(img_tensor) * noise_magnitude
    noisy_img_tensor = img_tensor + noise
    
    # clip the pixel values to the valid range [0, 1]
    noisy_img_tensor = torch.clamp(noisy_img_tensor, 0, 1)

    return noisy_img_tensor

if args.dataset == 'cifar10' or args.dataset == 'gtsrb':

    # randomly sample from a clean test set to simulate the clean samples at hand
    num_img = len(clean_set)
    id_set = list(range(0, num_img))
    random.shuffle(id_set)
    clean_split_indices = id_set[:args.clean_budget]
    test_indices = id_set[args.clean_budget:]


    # Take the rest clean samples as the test set for debug & evaluation
    test_set = torch.utils.data.Subset(clean_set, test_indices)
    num = len(test_set)
    label_set = []

    for i in range(num):
        img, gt = test_set[i]
        img_file_name = '%d.png' % i
        img_file_path = os.path.join(test_split_img_dir, img_file_name)
        save_image(img, img_file_path)
        print('[Generate Noisy Test Set] Save %s' % img_file_path)
        label_set.append(gt)
    
    for i in range(num):
        img, gt = test_set[i]
        img_noise = AddNoise(img)
        img_file_name = '%d.png' % (i + num)
        img_file_path = os.path.join(test_split_img_dir, img_file_name)
        save_image(img_noise, img_file_path)
        print('[Generate Noisy Test Set] Save %s' % img_file_path)
        label_set.append(gt)

    label_set = torch.LongTensor(label_set)
    label_path = os.path.join(test_split_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Test Set] Save %s' % label_path)

else: raise NotImplementedError