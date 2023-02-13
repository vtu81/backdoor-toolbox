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


datasets.ImageNet



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
elif args.dataset == 'imagenette':

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    clean_set = datasets.ImageFolder(os.path.join(os.path.join(data_dir, 'imagenette2'), 'val'), data_transform)

    img_size = 224
    num_classes = 10

elif args.dataset == 'imagenet':
    pass

elif args.dataset == 'ember':

    import ember

    EMBER_DATA_DIR = os.path.join(data_dir, 'ember')

    # Perform feature vectorization only if necessary.
    try:
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            EMBER_DATA_DIR,
            feature_version=1
        )

    except:
        ember.create_vectorized_features(
            EMBER_DATA_DIR,
            feature_version=1
        )
        x_train, y_train, x_test, y_test = ember.read_vectorized_features(
            EMBER_DATA_DIR,
            feature_version=1
        )

    #x_train = x_train.astype(dtype='float64')
    x_test = x_test.astype(np.float)
    y_test = y_test.astype(np.long)

    # Get rid of unknown labels
    #x_train = x_train[y_train != -1]
    #y_train = y_train[y_train != -1]
    x_test = x_test[y_test != -1]
    y_test = y_test[y_test != -1]


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



if args.dataset != 'ember' and args.dataset != 'imagenet':

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

elif args.dataset == 'imagenet':

    # randomly sample from a clean test set to simulate the clean samples at hand
    num_img = 50000
    id_set = list(range(0, num_img))
    random.shuffle(id_set)
    clean_split_indices = id_set[:args.clean_budget]
    test_indices = id_set[args.clean_budget:]

    print('[Generate Clean Split Set] Save %s' % os.path.join(clean_split_dir, 'clean_split_indices'))
    torch.save(clean_split_indices, os.path.join(clean_split_dir, 'clean_split_indices') )

    print('[Generate Test Set] Save %s' % os.path.join(test_split_dir, 'test_indices'))
    torch.save(test_indices, os.path.join(test_split_dir, 'test_indices'))

else:

    num_samples = len(y_test)
    id_set = list(range(0, num_samples))
    random.shuffle(id_set)
    clean_split_indices = id_set[:args.clean_budget]
    test_indices = id_set[args.clean_budget:]


    x_clean_split = x_test[clean_split_indices]
    y_clean_split = y_test[clean_split_indices]

    x_test_split = x_test[test_indices]
    y_test_split = y_test[test_indices]

    np.save(os.path.join(clean_split_dir, 'X'), x_clean_split)
    np.save(os.path.join(clean_split_dir, 'Y'), y_clean_split)
    print('[Generate Clean Split Set] %s, %s' % (os.path.join(clean_split_dir, 'X'), os.path.join(clean_split_dir, 'Y')) )

    np.save(os.path.join(test_split_dir, 'X'), x_test_split)
    np.save(os.path.join(test_split_dir, 'Y'), y_test_split)
    print('[Generate Test Set] %s, %s' % (os.path.join(test_split_dir, 'X'), os.path.join(test_split_dir, 'Y')) )