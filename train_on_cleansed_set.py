import argparse
import os, sys
from tqdm import tqdm
import config
from torchvision import datasets, transforms
from torch import nn
import torch
from utils import default_args, supervisor, tools, imagenet
import time

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
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-cleanser', type=str, choices=['SCAn','AC','SS', 'CT', 'SPECTRE', 'Strip'], default='CT')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
# tools.setup_seed(args.seed)

if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]


all_to_all = False
if args.poison_type == 'badnet_all_to_all':
    all_to_all = True

if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, args.cleanser)
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_aug.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed)))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

batch_size = 128
kwargs = {'num_workers': 2, 'pin_memory': True}



if args.dataset == 'cifar10':

    num_classes = 10

    data_transform_aug = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
                                ])

    data_transform_no_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    ])

    trigger_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
    ])

    momentum = 0.9
    weight_decay = 1e-4
    milestones = [100, 150]
    epochs = 200
    learning_rate = 0.1

elif args.dataset == 'gtsrb':

    num_classes = 43

    data_transform_aug = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    data_transform_no_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    trigger_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    momentum = 0.9
    weight_decay = 1e-4
    epochs = 100
    milestones = torch.tensor([40, 80])
    learning_rate = 0.1

elif args.dataset == 'ember':

    num_classes = 2
    arch = supervisor.get_arch(args)
    momentum = 0.9
    weight_decay = 1e-6
    epochs = 10
    learning_rate = 0.1
    milestones = torch.tensor([])
    batch_size = 512

    print('[Non-image Dataset] Amber')

else:
    raise Exception("Invalid Dataset")



if args.dataset != 'ember':
    poison_set_dir = supervisor.get_poison_set_dir(args)
    if os.path.exists(os.path.join(poison_set_dir, 'data')): # if old version
        poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    if os.path.exists(os.path.join(poison_set_dir, 'imgs')): # if new version
        poisoned_set_img_dir = os.path.join(poison_set_dir, 'imgs')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                         label_path=poisoned_set_label_path, transforms=data_transform_aug)
    cleansed_set_indices_dir = supervisor.get_cleansed_set_indices_dir(args)
    print('load : %s' % cleansed_set_indices_dir)
    cleansed_set_indices = torch.load(cleansed_set_indices_dir)
else:
    poison_set_dir = os.path.join('poisoned_train_set', 'ember', args.ember_options)
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

    # stats_path = os.path.join('data', 'ember', 'stats')
    poisoned_set = tools.EMBER_Dataset(x_path=os.path.join(poison_set_dir, 'watermarked_X.npy'),
                                       y_path=os.path.join(poison_set_dir, 'watermarked_y.npy'))
    cleansed_set_indices_dir = os.path.join(poison_set_dir, 'cleansed_set_indices_seed=%d' % args.seed)
    print('load : %s' % cleansed_set_indices_dir)
    cleansed_set_indices = torch.load(cleansed_set_indices_dir)


poisoned_indices = torch.load(os.path.join(poison_set_dir, 'poison_indices'))
cleansed_set_indices.sort()
poisoned_indices.sort()

tot_poison = len(poisoned_indices)
num_poison = 0

if tot_poison > 0:
    pt = 0
    for pid in cleansed_set_indices:
        while poisoned_indices[pt] < pid and pt + 1 < tot_poison: pt += 1
        if poisoned_indices[pt] == pid:
            num_poison += 1

print('remaining poison samples in cleansed set : ', num_poison)



cleansed_set = torch.utils.data.Subset(poisoned_set, cleansed_set_indices)
train_set = cleansed_set


if args.dataset != 'ember':

    # Set Up Test Set for Debug & Evaluation
    test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
    test_set_img_dir = os.path.join(test_set_dir, 'data')
    test_set_label_path = os.path.join(test_set_dir, 'labels')
    test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                 label_path=test_set_label_path, transforms=data_transform_no_aug)
    test_set_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size, shuffle=False, worker_init_fn=tools.worker_init, **kwargs)

    # Poison Transform for Testing
    poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=config.target_class[args.dataset], trigger_transform=trigger_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)

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




train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size, shuffle=True, worker_init_fn=tools.worker_init, **kwargs)

arch = supervisor.get_arch(args)


if args.poison_type == 'TaCT':
    source_classes = [config.source_class]
else:
    source_classes = None

model = arch(num_classes=num_classes)
model = nn.DataParallel(model)
model = model.cuda()


if args.dataset != 'ember':

    print(f"Will save to '{supervisor.get_model_dir(args, cleanse=True)}'.")
    if os.path.exists(supervisor.get_model_dir(args, cleanse=True)):  # exit if there is an already trained model
        print(f"Model '{supervisor.get_model_dir(args, cleanse=True)}' already exists!")
        model = arch(num_classes=num_classes)
        model.load_state_dict(torch.load(supervisor.get_model_dir(args, cleanse=True)))
        model = model.cuda()
        tools.test(model=model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform,
                   num_classes=num_classes, source_classes=source_classes)
        exit(0)
    criterion = nn.CrossEntropyLoss().cuda()
else:
    model_path = os.path.join('poisoned_train_set', 'ember', args.ember_options, 'model_trained_on_cleansed_data_seed=%d.pt' % args.seed)
    print(f"Will save to '{model_path}'.")
    if os.path.exists(model_path):
        print(f"Model '{model_path}' already exists!")
    criterion = nn.BCELoss().cuda()


optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)
from tqdm import tqdm
for epoch in range(1,epochs+1):
    start_time = time.perf_counter()

    model.train()
    for data, target in tqdm(train_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()  # train set batch
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    scheduler.step()
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print('<Cleansed Training> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}, Time: {:.2f}s'.format(epoch, loss.item(), optimizer.param_groups[0]['lr'], elapsed_time))

    # Test
    if args.dataset != 'ember':
        if epoch % 20 == 0:
            tools.test(model=model, test_loader=test_set_loader, poison_test=True,
                           poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes)
            torch.save(model.module.state_dict(), supervisor.get_model_dir(args, cleanse=True))
    else:
        if epoch % 5 == 0:
            tools.test_ember(model=model, test_loader=test_set_loader, backdoor_test_loader=backdoor_test_set_loader)
            torch.save(model.module.state_dict(), model_path)

if args.dataset != 'ember':
    torch.save(model.module.state_dict(), supervisor.get_model_dir(args, cleanse=True))
else:
    torch.save(model.module.state_dict(), model_path)