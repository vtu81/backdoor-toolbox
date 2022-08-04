import os
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import argparse
import random
import config

"""
<Four Data Sets>
GTSRB, CIFAR10, CIFAR100, Imagenette (imagenet subset)
"""

torch.manual_seed(666)
random.seed(666)


parser = argparse.ArgumentParser()

parser.add_argument('-dataset', type=str, required=False, default=config.parser_default['dataset'],
                    choices=config.parser_choices['dataset'])
parser.add_argument('-clean_budget', type=int, default=2000)
# by defaut :  we assume 2000 clean samples for defensive purpose

args = parser.parse_args()


"""
Get Data Set
"""
data_dir = './data' # directory to save standard clean set

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
elif args.dataset == 'cifar100':
    print('<To Be Implemented> Dataset = %s' % args.dataset)
    exit(0)
elif args.dataset == 'imagenette':
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    if not os.path.exists(os.path.join(os.path.join(data_dir, 'imagenette2'), 'val')):
        print("Please download full size Imagenette dataset from https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz and extract it under ./data/")
    clean_set = datasets.ImageFolder(os.path.join(os.path.join(data_dir, 'imagenette2'), 'val'), data_transform)
    
    img_size = 224
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

clean_split_dir = os.path.join(root_dir, 'clean_split') # clean samples at hand for defensive purpose
if not os.path.exists(clean_split_dir):
    os.mkdir(clean_split_dir)

clean_split_img_dir = os.path.join(clean_split_dir, 'data') # to save img
if not os.path.exists(clean_split_img_dir):
    os.mkdir(clean_split_img_dir)

test_split_dir = os.path.join(root_dir, 'test_split') # test samples for evaluation & debug purpose
if not os.path.exists(test_split_dir):
    os.mkdir(test_split_dir)

test_split_img_dir = os.path.join(test_split_dir, 'data') # to save img
if not os.path.exists(test_split_img_dir):
    os.mkdir(test_split_img_dir)


# randomly sample from a clean test set to simulate the clean samples at hand
num_img = len(clean_set)
id_set = list(range(0,num_img))
random.shuffle(id_set)
clean_split_indices = id_set[:args.clean_budget]
test_indices = id_set[args.clean_budget:]

# Construct Shift Set for Defensive Purpose
clean_split_set = torch.utils.data.Subset(clean_set, clean_split_indices)
num = len(clean_split_set)

clean_label_set = []

for i in range(num):
    img, gt = clean_split_set[i]
    img_file_name = '%d.png' % i
    img_file_path = os.path.join(clean_split_img_dir, img_file_name)
    save_image(img, img_file_path)
    print('[Generate Clean Split] Save %s' % img_file_path)
    clean_label_set.append(gt)

clean_label_set = torch.LongTensor(clean_label_set)
clean_label_path = os.path.join(clean_split_dir, 'clean_labels')
torch.save(clean_label_set, clean_label_path)
print('[Generate Clean Split Set] Save %s' % clean_label_path)




# Take the rest clean samples as the test set for debug & evaluation
test_set = torch.utils.data.Subset(clean_set, test_indices)
num = len(test_set)
label_set = []

for i in range(num):
    img, gt = test_set[i]
    img_file_name = '%d.png' % i
    img_file_path = os.path.join(test_split_img_dir, img_file_name)
    save_image(img, img_file_path)
    print('[Generate Test Set] Save %s' % img_file_path)
    label_set.append(gt)

label_set = torch.LongTensor(label_set)
label_path = os.path.join(test_split_dir, 'labels')
torch.save(label_set, label_path)
print('[Generate Test Set] Save %s' % label_path)