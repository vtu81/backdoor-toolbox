import os
import torch
from torchvision import datasets, transforms
import argparse
from PIL import Image
import numpy as np
import config
from utils import supervisor

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
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
args = parser.parse_args()

print('[target class : %d]' % config.target_class[args.dataset])

data_dir = config.data_dir  # directory to save standard clean set
if args.trigger is None:
    args.trigger = config.trigger_default[args.poison_type]

if not os.path.exists(os.path.join('poisoned_train_set', args.dataset)):
    os.mkdir(os.path.join('poisoned_train_set', args.dataset))

if args.poison_type == 'dynamic':

    if args.dataset == 'cifar10':

        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                     download=True, transform=data_transform)
        img_size = 32
        num_classes = 10
        channel_init = 32
        steps = 3
        input_channel = 3

        ckpt_path = './models/all2one_cifar10_ckpt.pth.tar'

        normalizer = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])

        denormalizer = transforms.Compose([
            transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261], [1 / 0.247, 1 / 0.243, 1 / 0.261])
        ])

    elif args.dataset == 'gtsrb':

        data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split='train',
                                   transform=data_transform, download=True)

        img_size = 32
        num_classes = 43
        channel_init = 32
        steps = 3
        input_channel = 3

        ckpt_path = './models/all2one_gtsrb_ckpt.pth.tar'

        normalizer = None
        denormalizer = None

    elif args.dataset == 'cifar100':
        raise  NotImplementedError('cifar100 unsupported!')
    elif args.dataset == 'imagenette':
        raise  NotImplementedError('imagenette unsupported!')
    else:
        raise  NotImplementedError('Undefined Dataset')

elif args.poison_type == 'ISSBA':

    if args.dataset == 'cifar10':

        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                     download=True, transform=data_transform)
        img_size = 32
        num_classes = 10
        input_channel = 3

        ckpt_path = './models/ISSBA_cifar10.pth'

    elif args.dataset == 'gtsrb':

        data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split='train',
                                   transform=data_transform, download=True)

        img_size = 32
        num_classes = 43
        input_channel = 3

        ckpt_path = './models/ISSBA_gtsrb.pth'

    elif args.dataset == 'cifar100':
        raise  NotImplementedError('cifar100 unsupported!')
    elif args.dataset == 'imagenette':
        raise  NotImplementedError('imagenette unsupported!')
    else:
        raise  NotImplementedError('Undefined Dataset')

else:
    if args.dataset == 'gtsrb':
        data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_set = datasets.GTSRB(os.path.join(data_dir, 'gtsrb'), split = 'train',
                                   transform = data_transform, download=True)
        img_size = 32
        num_classes = 43
    elif args.dataset == 'cifar10':
        data_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_set = datasets.CIFAR10(os.path.join(data_dir, 'cifar10'), train=True,
                                     download=True, transform=data_transform)
        img_size = 32
        num_classes = 10
    elif args.dataset == 'cifar100':
        raise NotImplementedError('cifar100 unsupported!')
    elif args.dataset == 'imagenette':
        data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        train_set = datasets.ImageFolder(os.path.join(os.path.join(data_dir, 'imagenette2'), 'train'), data_transform)

        img_size = 224
        num_classes = 10
        trigger_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        raise  NotImplementedError('Undefined Dataset')

trigger_transform = transforms.Compose([
    transforms.ToTensor()
])

# Create poisoned dataset directory for current setting
poison_set_dir = supervisor.get_poison_set_dir(args)
poison_set_img_dir = os.path.join(poison_set_dir, 'data')

if not os.path.exists(poison_set_dir):
    os.mkdir(poison_set_dir)
if not os.path.exists(poison_set_img_dir):
    os.mkdir(poison_set_img_dir)



if args.poison_type in ['basic', 'badnet', 'blend', 'clean_label', 'refool',
                        'adaptive', 'adaptive_blend', 'adaptive_k', 'adaptive_k_way', 'adaptive_mask',
                        'SIG', 'TaCT', 'WaNet', 'none']:

    trigger_name = args.trigger
    trigger_path = os.path.join(config.triggers_dir, trigger_name)

    trigger = None
    trigger_mask = None

    if trigger_name != 'none':  # none for SIG

        print('trigger: %s' % trigger_path)

        trigger_path = os.path.join(config.triggers_dir, trigger_name)
        trigger = Image.open(trigger_path).convert("RGB")
        trigger = trigger_transform(trigger)

        trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_name)
        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            #print('trigger_mask_path:', trigger_mask_path)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            #print('No trigger mask found! By default masking all black pixels...')
            trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0), trigger[2] > 0).float()

    alpha = args.alpha

    poison_generator = None

    if args.poison_type == 'basic':

        from poison_tool_box import basic
        poison_generator = basic.poison_generator(img_size=img_size, dataset=train_set,
                                                  poison_rate=args.poison_rate,
                                                  path=poison_set_img_dir,
                                                  trigger_mark=trigger, trigger_mask=trigger_mask,
                                                  target_class=config.target_class[args.dataset], alpha=alpha)

    elif args.poison_type == 'badnet':

        from poison_tool_box import badnet
        poison_generator = badnet.poison_generator(img_size=img_size, dataset=train_set,
                                                   poison_rate=args.poison_rate, trigger=trigger,
                                                   path=poison_set_img_dir, target_class=config.target_class[args.dataset])

    elif args.poison_type == 'blend':

        from poison_tool_box import blend
        poison_generator = blend.poison_generator(img_size=img_size, dataset=train_set,
                                                  poison_rate=args.poison_rate, trigger=trigger,
                                                  path=poison_set_img_dir, target_class=config.target_class[args.dataset],
                                                  alpha=alpha)
    elif args.poison_type == 'refool':
        from poison_tool_box import refool
        poison_generator = refool.poison_generator(img_size=img_size, dataset=train_set,
                                                  poison_rate=args.poison_rate,
                                                  path=poison_set_img_dir, target_class=config.target_class[args.dataset],
                                                  max_image_size=32)

    elif args.poison_type == 'TaCT':

        from poison_tool_box import TaCT
        poison_generator = TaCT.poison_generator(img_size=img_size, dataset=train_set,
                                                 poison_rate=args.poison_rate, cover_rate=args.cover_rate,
                                                 trigger=trigger, mask=trigger_mask,
                                                 path=poison_set_img_dir, target_class=config.target_class[args.dataset],
                                                 source_class=config.source_class,
                                                 cover_classes=config.cover_classes)

    elif args.poison_type == 'WaNet':

        from poison_tool_box import WaNet
        poison_generator = WaNet.poison_generator(img_size=img_size, dataset=train_set,
                                                 poison_rate=args.poison_rate, cover_rate=args.cover_rate,
                                                 path=poison_set_img_dir, target_class=config.target_class[args.dataset])

    elif args.poison_type == 'adaptive':

        from poison_tool_box import adaptive
        poison_generator = adaptive.poison_generator(img_size=img_size, dataset=train_set,
                                                     poison_rate=args.poison_rate,
                                                     path=poison_set_img_dir,
                                                     trigger_mark=trigger, trigger_mask=trigger_mask,
                                                     target_class=config.target_class[args.dataset], alpha=alpha,
                                                     cover_rate=args.cover_rate)
    
    elif args.poison_type == 'adaptive_blend':

        from poison_tool_box import adaptive_blend
        poison_generator = adaptive_blend.poison_generator(img_size=img_size, dataset=train_set,
                                                           poison_rate=args.poison_rate,
                                                           path=poison_set_img_dir, trigger=trigger,
                                                           target_class=config.target_class[args.dataset], alpha=alpha,
                                                           cover_rate=args.cover_rate)

    elif args.poison_type == 'adaptive_mask':

        from poison_tool_box import adaptive_mask
        poison_generator = adaptive_mask.poison_generator(img_size=img_size, dataset=train_set,
                                                           poison_rate=args.poison_rate,
                                                           path=poison_set_img_dir, trigger=trigger,
                                                           pieces=16, mask_rate=0.5,
                                                           target_class=config.target_class[args.dataset], alpha=alpha,
                                                           cover_rate=args.cover_rate)
    
    elif args.poison_type == 'adaptive_k_way':

        from poison_tool_box import adaptive_k_way
        poison_generator = adaptive_k_way.poison_generator(img_size=img_size, dataset=train_set,
                                                           poison_rate=args.poison_rate,
                                                           path=poison_set_img_dir,
                                                           target_class=config.target_class[args.dataset],
                                                           cover_rate=args.cover_rate)
    
    elif args.poison_type == 'adaptive_k':

        from poison_tool_box import adaptive_k
        poison_generator = adaptive_k.poison_generator(img_size=img_size, dataset=train_set,
                                                       poison_rate=args.poison_rate,
                                                       path=poison_set_img_dir,
                                                       target_class=config.target_class[args.dataset],
                                                       cover_rate=args.cover_rate)

    elif args.poison_type == 'SIG':

        from poison_tool_box import SIG
        poison_generator = SIG.poison_generator(img_size=img_size, dataset=train_set,
                                                poison_rate=args.poison_rate,
                                                path=poison_set_img_dir, target_class=config.target_class[args.dataset],
                                                delta=30/255, f=6)

    elif args.poison_type == 'clean_label':

        if args.dataset == 'cifar10':
            adv_imgs_path = "data/cifar10/clean_label/fully_poisoned_training_datasets/two_600.npy"
            if not os.path.exists("data/cifar10/clean_label/fully_poisoned_training_datasets/two_600.npy"):
                raise NotImplementedError("Run 'data/cifar10/clean_label/setup.sh' first to launch clean label attack!")
            adv_imgs_src = np.load("data/cifar10/clean_label/fully_poisoned_training_datasets/two_600.npy").astype(
                np.uint8)
            adv_imgs = []
            for i in range(adv_imgs_src.shape[0]):
                adv_imgs.append(data_transform(adv_imgs_src[i]).unsqueeze(0))
            adv_imgs = torch.cat(adv_imgs, dim=0)
            assert adv_imgs.shape[0] == len(train_set)
        else:
            raise NotImplementedError('Clean Label Attack is not implemented for %s' % args.dataset)

        # Init Attacker
        from poison_tool_box import clean_label
        poison_generator = clean_label.poison_generator(img_size=img_size, dataset=train_set, adv_imgs=adv_imgs,
                                                        poison_rate=args.poison_rate,
                                                        trigger_mark = trigger, trigger_mask=trigger_mask,
                                                        path=poison_set_img_dir, target_class=config.target_class[args.dataset])

    else: # 'none'
        from poison_tool_box import none
        poison_generator = none.poison_generator(img_size=img_size, dataset=train_set,
                                                path=poison_set_img_dir)



    if args.poison_type not in ['TaCT', 'WaNet', 'adaptive', 'adaptive_blend', 'adaptive_mask', 'adaptive_k', 'adaptive_k_way']:
        poison_indices, label_set = poison_generator.generate_poisoned_training_set()
        print('[Generate Poisoned Set] Save %d Images' % len(label_set))

    else:
        poison_indices, cover_indices, label_set = poison_generator.generate_poisoned_training_set()
        print('[Generate Poisoned Set] Save %d Images' % len(label_set))

        cover_indices_path = os.path.join(poison_set_dir, 'cover_indices')
        torch.save(cover_indices, cover_indices_path)
        print('[Generate Poisoned Set] Save %s' % cover_indices_path)



    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)

    #print('poison_indices : ', poison_indices)


elif args.poison_type == 'dynamic':
    """
        Since we will use the pretrained model by the original paper, here we use normalized data following 
        the original implementation.
        Download Pretrained Generator from https://github.com/VinAIResearch/input-aware-backdoor-attack-release
    """
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('[Dynamic Attack] Download pretrained generator first : https://github.com/VinAIResearch/input-aware-backdoor-attack-release')
    # Init Attacker
    from poison_tool_box import dynamic
    poison_generator = dynamic.poison_generator(ckpt_path=ckpt_path, channel_init=channel_init, steps=steps,
                                                input_channel=input_channel, normalizer=normalizer,
                                                denormalizer=denormalizer, dataset=train_set,
                                                poison_rate=args.poison_rate, path=poison_set_img_dir, target_class=config.target_class[args.dataset])

    # Generate Poison Data
    poison_indices, label_set = poison_generator.generate_poisoned_training_set()
    print('[Generate Poisoned Set] Save %d Images' % len(label_set))

    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)

elif args.poison_type == 'ISSBA':
    # if not os.path.exists(ckpt_path):
    #     raise NotImplementedError('[ISSBA Attack] Download pretrained encoder and decoder first: https://github.com/')
    
    # Init Secret
    secret_size = 20
    secret = torch.FloatTensor(np.random.binomial(1, .5, secret_size).tolist())
    secret_path = os.path.join(poison_set_dir, 'secret')
    torch.save(secret, secret_path)
    print('[Generate Poisoned Set] Save %s' % secret_path)
    
    # Init Attacker
    from poison_tool_box import ISSBA
    poison_generator = ISSBA.poison_generator(ckpt_path=ckpt_path, secret=secret, dataset=train_set, enc_height=img_size, enc_width=img_size, enc_in_channel=input_channel,
                                                poison_rate=args.poison_rate, path=poison_set_img_dir, target_class=config.target_class[args.dataset])

    # Generate Poison Data
    poison_indices, label_set = poison_generator.generate_poisoned_training_set()
    print('[Generate Poisoned Set] Save %d Images' % len(label_set))

    label_path = os.path.join(poison_set_dir, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Poisoned Set] Save %s' % label_path)

    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    torch.save(poison_indices, poison_indices_path)
    print('[Generate Poisoned Set] Save %s' % poison_indices_path)

else:
    raise NotImplementedError('%s not defined' % args.poison_type)