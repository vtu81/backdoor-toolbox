import torch
import os
from utils import resnet, tools
from torchvision import transforms
import argparse
from torch import nn
from tqdm import tqdm
import torchvision
import config

parser = argparse.ArgumentParser()
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-dataset', type=str, required=False,
                    default=config.parser_default['dataset'],
                    choices=config.parser_choices['dataset'])
parser.add_argument('-epoch', type=int, required=False,
                    choices=[40, 80, 200], default=200)
parser.add_argument('-seed', type=int, required=False, default=config.seed)
args = parser.parse_args()
tools.setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices


data_transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
])

data_transform_no_aug = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
])

data_transform = data_transform_no_aug if args.no_aug else data_transform_aug


batch_size = 256
learning_rate = 0.1
num_classes = 10
arch = resnet.resnet110
momentum = 0.9
weight_decay = 1e-4
milestones = torch.tensor([100, 150])
n_epochs = 200

trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True,
                                        download=True, transform=data_transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                       download=True, transform=data_transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=8)

if args.dataset == 'cifar10':
    num_classes = 10
    arch = config.arch[args.dataset]
    momentum = 0.9
    weight_decay = 1e-4
    if args.epoch == 200: milestones = torch.tensor([100, 150])
    elif args.epoch == 80: milestones = torch.tensor([30, 50])
    elif args.epoch == 40: milestones = torch.tensor([15, 30])
    backdoor_epochs = args.epoch
    learning_rate = 0.1

elif args.dataset == 'cifar100':
    num_classes = 100
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)
elif args.dataset == 'gtsrb':
    num_classes = 43
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)
elif args.dataset == 'imagenette':
    num_classes = 10
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)
else:
    print('<Undefined Dataset> Dataset = %s' % args.dataset)
    raise NotImplementedError('<To Be Implemented> Dataset = %s' % args.dataset)


# Train Code
model = arch(num_classes=num_classes)
milestones = milestones.tolist()
model = nn.DataParallel(model)
model = model.cuda()

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones)

for epoch in range(1, n_epochs + 1):  # train backdoored base model
    # Train
    model.train()
    for data, target in tqdm(train_loader):
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()  # train set batch
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print('<Vanilla Training> Train Epoch: {} \tLoss: {:.6f}, lr: {}'.format(epoch, loss.item(), optimizer.param_groups[0]['lr']))
    scheduler.step()

    tools.test(model=model, test_loader=test_loader)
    if args.no_aug:
        torch.save(model.module.state_dict(), 'models/%s_vanilla_no_aug.pt' % args.dataset)
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug_seed={args.seed}.pt')
    else:
        torch.save(model.module.state_dict(), 'models/%s_vanilla_aug.pt' % args.dataset)
        torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug_seed={args.seed}.pt')


if args.no_aug:
    torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug.pt')
    torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_no_aug_seed={args.seed}.pt')
else:
    torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug.pt')
    torch.save(model.module.state_dict(), f'models/{args.dataset}_vanilla_aug_seed={args.seed}.pt')
print('[Done]')