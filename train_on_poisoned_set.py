import torch
import os, sys
from torchvision import transforms
import argparse
from torch import nn
from utils import supervisor, tools
import config
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=config.parser_default['dataset'],
                    choices=config.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=True,
                    choices=config.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=config.parser_choices['poison_rate'],
                    default=config.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=config.parser_choices['cover_rate'],
                    default=config.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float, required=False,
                    default=config.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=config.seed)

args = parser.parse_args()

if args.trigger is None:
    args.trigger = config.trigger_default[args.poison_type]

tools.setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'base')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_%s.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed), 'no_aug' if args.no_aug else 'aug'))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

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

elif args.dataset == 'imagenette':

    data_transform_aug = transforms.Compose([
        transforms.RandomCrop(224, 4),
        transforms.RandomHorizontalFlip(),  
        transforms.ColorJitter(brightness=0.4, contrast=0.4,saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

else:
    raise NotImplementedError('dataset %s not supported' % args.dataset)




batch_size = 128

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
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 200
    milestones = torch.tensor([100, 150])
    learning_rate = 0.1
else:
    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)


kwargs = {'num_workers': 2, 'pin_memory': True}

# Set Up Poisoned Set
poison_set_dir = supervisor.get_poison_set_dir(args)

poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                 label_path=poisoned_set_label_path, transforms=data_transform if args.no_aug else data_transform_aug)

poisoned_set_loader = torch.utils.data.DataLoader(
    poisoned_set,
    batch_size=batch_size, shuffle=True, **kwargs)

poisoned_set_loader_no_shuffle = torch.utils.data.DataLoader(
    poisoned_set,
    batch_size=batch_size, shuffle=False, **kwargs)

poison_indices = torch.tensor(torch.load(poison_indices_path)).cuda()

# Set Up Test Set for Debug & Evaluation
test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
test_set_img_dir = os.path.join(test_set_dir, 'data')
test_set_label_path = os.path.join(test_set_dir, 'labels')
test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                             label_path=test_set_label_path, transforms=data_transform)
test_set_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size, shuffle=True, **kwargs)

# Poison Transform for Testing
poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                   target_class=config.target_class[args.dataset], trigger_transform=data_transform,
                                                   is_normalized_input=True,
                                                   alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                   trigger_name=args.trigger, args=args)

# Train Code
model = arch(num_classes=num_classes)
milestones = milestones.tolist()
model = nn.DataParallel(model)
model = model.cuda()
print(f"Will save to '{supervisor.get_model_dir(args)}'.")


if os.path.exists(supervisor.get_model_dir(args)): # exit if there is an already trained model
    print(f"Model '{supervisor.get_model_dir(args)}' already exists!")
    exit(0)


criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

if args.poison_type == 'TaCT':
    source_classes = [config.source_class]
else:
    source_classes = None

for epoch in range(1, epochs+1):  # train backdoored base model
    # Train
    model.train()
    preds = []
    labels = []
    for data, target in tqdm(poisoned_set_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()  # train set batch
        output = model(data)
        preds.append(output.argmax(dim=1))
        labels.append(target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    train_acc = (torch.eq(preds, labels).int().sum()) / preds.shape[0]
    print('\n<Backdoor Training> Train Epoch: {} \tLoss: {:.6f}, Train Acc: {:.6f}, lr: {:.2f}'.format(epoch, loss.item(), train_acc, optimizer.param_groups[0]['lr']))
    scheduler.step()

    if epoch % 20 == 0:
        # Test
        tools.test(model=model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes)
        torch.save(model.module.state_dict(), supervisor.get_model_dir(args))

torch.save(model.module.state_dict(), supervisor.get_model_dir(args))
if args.poison_type == 'none':
    if args.no_aug:
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug.pt')
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug_seed={args.seed}.pt')
    else:
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug.pt')
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug_seed={args.seed}.pt')