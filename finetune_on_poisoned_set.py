"""
Given a pretrained clean model, finetune it on a poisoned training subset.
This script is used to efficiently inject backdoor into large models, e.g. ViT.
"""

import argparse
import os, sys
import time
from tqdm import tqdm
from utils import default_args, imagenet
from torch.cuda.amp import autocast, GradScaler

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    default='none',
                    choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-ember_options', type=str, required=False,
                    choices=['constrained', 'unconstrained', 'none'],
                    default='unconstrained')
parser.add_argument('-alpha', type=float, required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-resume', type=int, required=False, default=0)
parser.add_argument('-resume_from_meta_info', default=False, action='store_true')
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices

import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import supervisor, tools


if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]


all_to_all = False
if args.poison_type == 'badnet_all_to_all':
    all_to_all = True


if args.dataset != 'ember':
    model_path = supervisor.get_model_dir(args)
else:
    model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options, 'backdoored_model.pt')

poison_set_dir = supervisor.get_poison_set_dir(args)
if not os.path.exists(poison_set_dir):
    os.makedirs(poison_set_dir)

# tools.setup_seed(args.seed)

if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'finetune')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_%s.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed), 'no_aug' if args.no_aug else 'aug'))
    if args.resume > 0 or args.resume_from_meta_info:
        fout = open(out_path, 'a')
    else:
        fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)


if args.dataset == 'cifar10':

    num_classes = 10
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([50, 75])
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'gtsrb':

    num_classes = 43
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([30, 60])
    learning_rate = 0.01
    batch_size = 128

elif args.dataset == 'imagenette':

    num_classes = 10
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([40, 80])
    learning_rate = 0.1
    batch_size = 128

elif args.dataset == 'imagenet':

    num_classes = 1000
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 5
    milestones = torch.tensor([30, 60])
    learning_rate = 0.01
    batch_size = 256

else:

    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)


if args.dataset == 'imagenet':
    kwargs = {'num_workers': 32, 'pin_memory': True}
else:
    kwargs = {'num_workers': 4, 'pin_memory': True}



if args.dataset != 'imagenet':

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    # Poison Transform for Testing
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=config.target_class[args.dataset], trigger_transform=data_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)


elif args.dataset == 'imagenet':

    # poison_transform = imagenet.get_poison_transform_for_imagenet(args.poison_type)
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=config.target_class[args.dataset], trigger_transform=data_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)

    test_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, data_transform=data_transform,
                 label_file=imagenet.test_set_labels, num_classes=1000)

    test_split_meta_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_indices = torch.load(os.path.join(test_split_meta_dir, 'test_indices'))

    test_set = torch.utils.data.Subset(test_set, test_indices)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

else:
    normalizer = poisoned_set.normal

    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')

    test_set = tools.EMBER_Dataset(x_path=os.path.join(test_set_dir, 'X.npy'),
                                   y_path=os.path.join(test_set_dir, 'Y.npy'),
                                   normalizer = normalizer)

    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)


    backdoor_test_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    backdoor_test_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X_test.npy'),
                                       y_path=None, normalizer = normalizer)
    backdoor_test_set_loader = torch.utils.data.DataLoader(
        backdoor_test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)



# Train Code
print(f"Will save to '{model_path}'.")
if os.path.exists(model_path):
    print(f"Model '{model_path}' already exists!")


if args.dataset == 'imagenet':
    # model = arch(num_classes=num_classes, weights='IMAGENET1K_V1')
    model = arch(num_classes=num_classes, weights='IMAGENET1K_SWAG_LINEAR_V1')
    # if 'vit' in arch.__name__:
    #     for param in model.encoder.parameters():
    #         param.requires_grad = False
elif args.dataset == 'cifar10' or args.dataset == 'gtsrb':
    model = arch(num_classes=num_classes)
    poison_type = args.poison_type
    poison_rate = args.poison_rate
    args.poison_type = 'none'
    args.poison_rate = 0
    clean_model_dir = supervisor.get_model_dir(args)
    model.load_state_dict(torch.load(clean_model_dir))
    args.poison_type = poison_type
    args.poison_rate = poison_rate
else:
    model = arch(num_classes=num_classes)

milestones = milestones.tolist()
model = nn.DataParallel(model)
model = model.cuda()

if args.poison_type == 'none':
    print(f"No poison is specified. Saved pretrained model to {model_path}!")
    torch.save(model.module.state_dict(), model_path)
    exit()

if args.dataset != 'ember':
    if args.dataset == 'imagenet':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.BCELoss().cuda()

optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

if args.poison_type == 'TaCT' or args.poison_type == 'SleeperAgent':
    source_classes = [config.source_class]
else:
    source_classes = None




"""
Finetuning dataset configs
"""
import random
from torch.utils.data import Dataset, random_split

from PIL import Image
trigger_name = args.trigger

# trigger mask transform; remove `Normalize`!
trigger_mask_transform_list = []
for t in trigger_transform.transforms:
    if "Normalize" not in t.__class__.__name__:
        trigger_mask_transform_list.append(t)
trigger_mask_transform = transforms.Compose(trigger_mask_transform_list)

if trigger_name != 'none': # none for SIG
    trigger_path = os.path.join(config.triggers_dir, trigger_name)
    # print('trigger : ', trigger_path)
    trigger = Image.open(trigger_path).convert("RGB")

    trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_name)

    if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
        trigger_mask = Image.open(trigger_mask_path).convert("RGB")
        trigger_mask = trigger_mask_transform(trigger_mask)[0]  # only use 1 channel
    else:  # by default, all black pixels are masked with 0's
        trigger_map = trigger_mask_transform(trigger)
        trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0), trigger_map[2] > 0).float()

    trigger = trigger_transform(trigger)
    trigger_mask = trigger_mask


if args.dataset == 'cifar10':
    ratio = 1.0
    poison_ratio = 0.2
    
    full_train_set = datasets.CIFAR10(root=os.path.join(config.data_dir, 'cifar10'), train=True, download=True, transform=data_transform_aug)
    batch_size = 128
    lr = 0.01
elif args.dataset == 'gtsrb':
    ratio = 1.0
    poison_ratio = 0.2
    
    full_train_set = datasets.GTSRB(os.path.join(config.data_dir, 'gtsrb'), split='train', download=True, transform=data_transform_aug)
    batch_size = 128
    lr = 0.001
elif args.dataset == 'imagenet':
    ratio = 0.1
    poison_ratio = 0.2
    
    from utils import imagenet
    train_set_dir = os.path.join(config.imagenet_dir, 'train')
    full_train_set = imagenet.imagenet_dataset(directory=train_set_dir, data_transform=data_transform_aug,
                                                poison_directory=None, poison_indices=None, target_class=config.target_class['imagenet'], num_classes=1000)
    batch_size = 256
    # lr = 0.002 # IMAGENET1K_V1
    lr = 0.00001 # IMAGENET1K_SWAG_LINEAR_V1
else:
    raise NotImplementedError()



from torch.utils.data import DataLoader, Dataset, Subset
id_set = list(range(0, len(full_train_set)))
random.shuffle(id_set)
finetune_indices = id_set[:int(len(id_set) * ratio)]
train_data = Subset(full_train_set, finetune_indices)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)



for epoch in range(1):  # train backdoored base model
    # Train
    model.train()
    preds = []
    labels = []
    for data, target in tqdm(train_loader):
        from torchvision.utils import save_image
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()  # train set batch
        
        id_set = list(range(0, len(data)))
        random.shuffle(id_set)
        poison_num = int(len(data) * poison_ratio)
        poison_set = id_set[:poison_num]
        data[poison_set], target[poison_set] = poison_transform.transform(data[poison_set], target[poison_set])
        
        # save_image(denormalizer(data), "a.png")
        # exit()
        
        output = model(data)
        preds.append(output.argmax(dim=1))
        labels.append(target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    train_acc = (torch.eq(preds, labels).int().sum()) / preds.shape[0]
    print('\n<Finetuning> Train Epoch: {} \tLoss: {:.6f}, Train Acc: {:.6f}, lr: {:.2f}'.format(epoch, loss.item(), train_acc, optimizer.param_groups[0]['lr']))
    tools.test(model=model, test_loader=test_set_loader, poison_test=True if args.poison_type != 'none' else False,
            poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes, all_to_all=all_to_all)

    torch.save(model.module.state_dict(), model_path)

torch.save(model.module.state_dict(), model_path)
