import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import argparse
import config

"""
In our defensive setting, we assume a poisoned training set and a small clean 
set at hand, i.e. we train base model jointly with the poisoned set and 
the shifted set (constructed based on the small clean set).

On the other hand, we also prepare a clean test set (usually larger than the 
small clean set used for defense in our experiments). But note that, this set is 
not used for defense, it is used for debug and evaluation!

Below we implement tools to take use of the additional clean test set for debug & evaluation.
"""


def get_cleansed_set_indices_dir(args):
    poison_set_dir = get_poison_set_dir(args)
    if args.cleanser == 'CT':  # confusion training
        return os.path.join(poison_set_dir, f'cleansed_set_indices_seed={args.seed}')
    else:
        return os.path.join(poison_set_dir, f'cleansed_set_indices_other_{args.cleanser}_seed={args.seed}')


def get_model_name(args, cleanse=False, defense=False):
    # `args.model_path` > `args.model` > by default 'full_base'
    if hasattr(args, 'model_path') and args.model_path is not None:
        model_name = args.model_path
    elif hasattr(args, 'model') and args.model is not None:
        model_name = args.model
    elif args.poison_type in ['trojannn']:
        model_name = f'{args.dataset}_{args.poison_type}_seed={args.seed}.pt'
    elif args.poison_type == 'SRA':
        model_name = f'{args.dataset}_{args.poison_type}_seed={args.seed}.pt'
    elif args.poison_type == 'BadEncoder':
        if args.dataset == 'gtsrb':
            model_name = 'BadEncoder_cifar2gtsrb.pth'
        else:
            raise NotImplementedError()
    else:
        if args.no_aug:
            model_name = f'full_base_no_aug_seed={args.seed}.pt'
        else:
            model_name = f'full_base_aug_seed={args.seed}.pt'

    if cleanse and hasattr(args, 'cleanser') and args.cleanser is not None:
        model_name = f"cleansed_{args.cleanser}_{model_name}"
    elif defense and hasattr(args, 'defense') and args.defense is not None:
        model_name = f"defended_{args.defense}_{model_name}"

    if config.record_model_arch:
        model_name = f"{get_arch(args).__name__}_{model_name}"
    return model_name


def get_model_dir(args, cleanse=False, defense=False):
    if hasattr(args, 'model_path') and args.model_path is not None:
        return args.model_path
    else:
        return f"{get_poison_set_dir(args)}/{get_model_name(args, cleanse=cleanse, defense=defense)}"


def get_dir_core(args, include_model_name=False, include_poison_seed=False):
    ratio = '%.3f' % args.poison_rate
    # ratio = '%.1f' % (args.poison_rate * 100) + '%'
    if args.poison_type in ['trojannn', 'BadEncoder', 'SRA']:
        dir_core = '%s_%s' % (args.dataset, args.poison_type)
    elif args.poison_type == 'blend' or args.poison_type == 'basic' or args.poison_type == 'clean_label':
        blend_alpha = '%.3f' % args.alpha
        dir_core = '%s_%s_%s_alpha=%s_trigger=%s' % (args.dataset, args.poison_type, ratio, blend_alpha, args.trigger)
    elif args.poison_type == 'adaptive_blend':
        blend_alpha = '%.3f' % args.alpha
        cover_rate = '%.3f' % args.cover_rate
        dir_core = '%s_%s_%s_alpha=%s_cover=%s_trigger=%s' % (
        args.dataset, args.poison_type, ratio, blend_alpha, cover_rate, args.trigger)
    elif args.poison_type == 'adaptive_patch' or args.poison_type == 'TaCT' or args.poison_type == 'WaNet':
        cover_rate = '%.3f' % args.cover_rate
        dir_core = '%s_%s_%s_cover=%s' % (args.dataset, args.poison_type, ratio, cover_rate)
    else:
        dir_core = '%s_%s_%s' % (args.dataset, args.poison_type, ratio)

    if include_model_name:
        dir_core = f'{dir_core}_{get_model_name(args)}'
    if include_poison_seed:
        dir_core = f'{dir_core}_poison_seed={config.poison_seed}'
    if config.record_model_arch:
        dir_core = f'{dir_core}_arch={get_arch(args).__name__}'
    return dir_core


def get_poison_set_dir(args):
    ratio = '%.3f' % args.poison_rate
    # ratio = '%.1f' % (args.poison_rate * 100) + '%'
    if args.poison_type in ['trojannn', 'BadEncoder', 'SRA']:
        poison_set_dir = 'models'
        return poison_set_dir
    elif args.poison_type == 'blend' or args.poison_type == 'basic' or args.poison_type == 'clean_label':
        blend_alpha = '%.3f' % args.alpha
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_alpha=%s_trigger=%s' % (
        args.dataset, args.poison_type, ratio, blend_alpha, args.trigger)
    elif args.poison_type == 'adaptive_blend':
        blend_alpha = '%.3f' % args.alpha
        cover_rate = '%.3f' % args.cover_rate
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_alpha=%s_cover=%s_trigger=%s' % (
        args.dataset, args.poison_type, ratio, blend_alpha, cover_rate, args.trigger)
    elif args.poison_type == 'adaptive_patch' or args.poison_type == 'TaCT' or args.poison_type == 'WaNet':
        cover_rate = '%.3f' % args.cover_rate
        poison_set_dir = 'poisoned_train_set/%s/%s_%s_cover=%s' % (args.dataset, args.poison_type, ratio, cover_rate)
    else:
        poison_set_dir = 'poisoned_train_set/%s/%s_%s' % (args.dataset, args.poison_type, ratio)

    if config.record_poison_seed: poison_set_dir = f'{poison_set_dir}_poison_seed={config.poison_seed}'  # debug
    # if config.record_model_arch: poison_set_dir = f'{poison_set_dir}_arch={get_arch(args).__name__}'
    return poison_set_dir


def get_arch(args):
    if args.poison_type == 'BadEncoder':
        if args.dataset == 'gtsrb':
            from utils.BadEncoder_model import CIFAR2GTSRB
            return CIFAR2GTSRB
        else:
            raise NotImplementedError
    if args.poison_type == 'SRA':
        if args.dataset == 'cifar10':
            if 'resnet' in config.arch[args.dataset].__name__.lower():
                from utils.SRA.cifar_10.resnet import resnet110
                return resnet110
            elif 'vgg' in config.arch[args.dataset].__name__:
                from utils.SRA.cifar_10.vgg import vgg16_bn
                return vgg16_bn
            elif 'mobilenet' in config.arch[args.dataset].__name__:
                from utils.SRA.cifar_10.mobilenetv2 import mobilenetv2
                return mobilenetv2
        elif args.dataset == 'imagenet':
            if 'vgg' in config.arch[args.dataset].__name__:
                from utils.SRA.imagenet.vgg import vgg16_bn
                return vgg16_bn
            elif 'resnet' in config.arch[args.dataset].__name__:
                from utils.SRA.imagenet.resnet import resnet101
                return resnet101
            elif 'mobilenetv2' in config.arch[args.dataset].__name__:
                from utils.SRA.imagenet.mobilenetv2 import mobilenet_v2
                return mobilenet_v2
        else:
            raise NotImplementedError
    if hasattr(args, 'defense') and args.defense == 'NONE':
        from other_defenses_tool_box.none.resnet import resnet18
        return resnet18
    else:
        return config.arch[args.dataset]


def get_transforms(args):
    if args.dataset == 'gtsrb':
        if args.no_normalize:
            data_transform_aug = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
            ])
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            data_transform_aug = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            data_transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            normalizer = transforms.Compose([
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
                                     (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
            ])

        if args.poison_type == 'BadEncoder':  # use CIFAR10's data transform for BadEncoder
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                     [1 / 0.247, 1 / 0.243, 1 / 0.261])
            ])

    elif args.dataset == 'cifar10':
        if args.no_normalize:
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor()
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                     [1 / 0.247, 1 / 0.243, 1 / 0.261])
            ])

        if args.poison_type == 'SRA':
            data_transform_aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                     [1 / 0.229, 1 / 0.224, 1 / 0.225])
            ])
    elif args.dataset == 'imagenette':
        if args.no_normalize:
            data_transform_aug = transforms.Compose([
                transforms.RandomCrop(224, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            normalizer = transforms.Compose([])
            denormalizer = transforms.Compose([])
        else:
            data_transform_aug = transforms.Compose([
                transforms.RandomCrop(224, 4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            trigger_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            normalizer = transforms.Compose([
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            denormalizer = transforms.Compose([
                transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                     [1 / 0.229, 1 / 0.224, 1 / 0.225])
            ])
    elif args.dataset == 'imagenet':
        data_transform_aug = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])
        data_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.CenterCrop(224),
        ])
        trigger_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.CenterCrop(224),
        ])
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        denormalizer = transforms.Compose([
            transforms.Normalize([-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], [1 / 0.229, 1 / 0.224, 1 / 0.225])
        ])

    elif args.dataset == 'ember':
        data_transform_aug = data_transform = trigger_transform = normalizer = denormalizer = None
    else:
        raise NotImplementedError()

    return data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer


def get_poison_transform(poison_type, dataset_name, target_class, source_class=1, cover_classes=[5, 7],
                         is_normalized_input=False, trigger_transform=None,
                         alpha=0.2, trigger_name=None, args=None):
    # source class will be used for TaCT poison

    if trigger_name is None:
        if dataset_name != 'imagenette':
            trigger_name = config.trigger_default[dataset_name][poison_type]
        else:
            if poison_type == 'badnet':
                trigger_name = 'badnet_high_res.png'
            else:
                raise NotImplementedError('%s not implemented for imagenette' % poison_type)

    if dataset_name in ['gtsrb', 'cifar10', 'cifar100']:
        img_size = 32
    elif dataset_name == 'imagenette' or dataset_name == 'imagenet':
        img_size = 224
    else:
        raise NotImplementedError('<Undefined> Dataset = %s' % dataset_name)

    if dataset_name == 'cifar10':
        normalizer = transforms.Compose([
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize([-0.4914 / 0.247, -0.4822 / 0.243, -0.4465 / 0.261],
                                 [1 / 0.247, 1 / 0.243, 1 / 0.261])
        ])
        num_classes = 10
    elif dataset_name == 'gtsrb':
        normalizer = transforms.Compose([
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
                                 (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
        ])
        num_classes = 43
    elif dataset_name == 'imagenette' or dataset_name == 'imagenet':
        normalizer = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        denormalizer = transforms.Compose([
            transforms.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
                                 (1.0 / 0.229, 1.0 / 0.224, 1.0 / 0.225)),
        ])
        num_classes = 10
    else:
        raise Exception("Invalid Dataset")

    poison_transform = None
    trigger = None
    trigger_mask = None

    if poison_type in ['basic', 'badnet', 'blend', 'clean_label', 'refool',
                       'adaptive_blend', 'adaptive_patch', 'adaptive_k_way',
                       'SIG', 'TaCT', 'WaNet', 'SleeperAgent', 'none',
                       'badnet_all_to_all', 'trojan', 'SRA', 'bpp']:

        if trigger_transform is None:
            trigger_transform = transforms.Compose([
                transforms.ToTensor()
            ])

        # trigger mask transform; remove `Normalize`!
        trigger_mask_transform_list = []
        for t in trigger_transform.transforms:
            if "Normalize" not in t.__class__.__name__:
                trigger_mask_transform_list.append(t)
        trigger_mask_transform = transforms.Compose(trigger_mask_transform_list)

        if trigger_name != 'none':  # none for SIG
            trigger_path = os.path.join(config.triggers_dir, trigger_name)
            # print('trigger : ', trigger_path)
            trigger = Image.open(trigger_path).convert("RGB")

            trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_name)

            if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = trigger_mask_transform(trigger_mask)[0]  # only use 1 channel
            else:  # by default, all black pixels are masked with 0's
                trigger_map = trigger_mask_transform(trigger)
                trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0),
                                                trigger_map[2] > 0).float()

            trigger = trigger_transform(trigger)
            # print('trigger_shape: ', trigger.shape)
            trigger_mask = trigger_mask

        if poison_type == 'basic':
            from poison_tool_box import basic
            poison_transform = basic.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                      trigger_mask=trigger_mask,
                                                      target_class=target_class, alpha=alpha)

        elif poison_type == 'badnet':
            from poison_tool_box import badnet
            poison_transform = badnet.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                       trigger_mask=trigger_mask, target_class=target_class)

        elif poison_type == 'badnet_all_to_all':
            from poison_tool_box import badnet_all_to_all
            poison_transform = badnet_all_to_all.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                                  trigger_mask=trigger_mask, num_classes=num_classes)

        elif poison_type == 'trojan':
            from poison_tool_box import trojan
            poison_transform = trojan.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                       trigger_mask=trigger_mask, target_class=target_class)

        elif poison_type == 'blend':
            from poison_tool_box import blend
            poison_transform = blend.poison_transform(img_size=img_size, trigger=trigger,
                                                      target_class=target_class, alpha=alpha)

        elif poison_type == 'refool':
            from poison_tool_box import refool
            poison_transform = refool.poison_transform(img_size=img_size, target_class=target_class,
                                                       denormalizer=denormalizer, normalizer=normalizer,
                                                       max_image_size=32)

        elif poison_type == 'clean_label':
            from poison_tool_box import clean_label
            poison_transform = clean_label.poison_transform(img_size=img_size, trigger_mark=trigger,
                                                            trigger_mask=trigger_mask,
                                                            target_class=target_class)

        elif poison_type == 'WaNet':
            s = 0.5
            k = 4
            grid_rescale = 1
            path = os.path.join(get_poison_set_dir(args), 'identity_grid')
            identity_grid = torch.load(path)
            path = os.path.join(get_poison_set_dir(args), 'noise_grid')
            noise_grid = torch.load(path)

            from poison_tool_box import WaNet
            poison_transform = WaNet.poison_transform(img_size=img_size, denormalizer=denormalizer,
                                                      identity_grid=identity_grid, noise_grid=noise_grid, s=s, k=k,
                                                      grid_rescale=grid_rescale, normalizer=normalizer,
                                                      target_class=target_class)

        elif poison_type == 'adaptive_blend':

            from poison_tool_box import adaptive_blend
            poison_transform = adaptive_blend.poison_transform(img_size=img_size, trigger=trigger,
                                                               target_class=target_class, alpha=alpha)

        elif poison_type == 'adaptive_patch':
            from poison_tool_box import adaptive_patch
            poison_transform = adaptive_patch.poison_transform(img_size=img_size, test_trigger_names=
            config.adaptive_patch_test_trigger_names[args.dataset],
                                                               test_alphas=config.adaptive_patch_test_trigger_alphas[
                                                                   args.dataset], target_class=target_class,
                                                               denormalizer=denormalizer, normalizer=normalizer, )

        elif poison_type == 'adaptive_k_way':
            from poison_tool_box import adaptive_k_way
            poison_transform = adaptive_k_way.poison_transform(img_size=img_size, target_class=target_class,
                                                               denormalizer=denormalizer, normalizer=normalizer, )

        elif poison_type == 'SIG':

            from poison_tool_box import SIG
            poison_transform = SIG.poison_transform(img_size=img_size, denormalizer=denormalizer, normalizer=normalizer,
                                                    target_class=target_class, delta=30 / 255, f=6,
                                                    has_normalized=is_normalized_input)

        elif poison_type == 'TaCT':
            from poison_tool_box import TaCT
            poison_transform = TaCT.poison_transform(img_size=img_size, trigger=trigger, mask=trigger_mask,
                                                     target_class=target_class)

        elif poison_type == 'SleeperAgent':
            from poison_tool_box import SleeperAgent
            poison_transform = SleeperAgent.poison_transform(random_patch=False, img_size=img_size,
                                                             target_class=target_class, denormalizer=denormalizer,
                                                             normalizer=normalizer)

        elif poison_type == 'SRA':
            if dataset_name not in ['cifar10', 'imagenet']:
                raise NotImplementedError()

            from other_attacks_tool_box import SRA
            poison_transform = SRA.poison_transform(img_size=img_size, trigger=trigger, mask=trigger_mask,
                                                    target_class=target_class)
            return poison_transform

        elif poison_type == 'bpp':
            if dataset_name not in ['cifar10']:
                raise NotImplementedError()

            from other_attacks_tool_box import bpp
            poison_transform = bpp.poison_transform(img_size=img_size, denormalizer=denormalizer, normalizer=normalizer,
                                                    mode="all2one", dithering=True, squeeze_num=8,
                                                    num_classes=num_classes, target_class=target_class)
            return poison_transform

        else:  # 'none'
            from poison_tool_box import none
            poison_transform = none.poison_transform()

        return poison_transform


    elif poison_type == 'dynamic':

        if dataset_name == 'cifar10':
            channel_init = 32
            steps = 3
            input_channel = 3
            ckpt_path = './models/all2one_cifar10_ckpt.pth.tar'

            require_normalization = True

        elif dataset_name == 'gtsrb':
            # the situation for gtsrb is inverese
            # the original implementation of generator does not require normalization
            channel_init = 32
            steps = 3
            input_channel = 3
            ckpt_path = './models/all2one_gtsrb_ckpt.pth.tar'

            require_normalization = False

        else:
            raise Exception("Invalid Dataset")

        if not os.path.exists(ckpt_path):
            raise NotImplementedError(
                '[Dynamic Attack] Download pretrained generator first: https://github.com/VinAIResearch/input-aware-backdoor-attack-release')

        from poison_tool_box import dynamic
        poison_transform = dynamic.poison_transform(ckpt_path=ckpt_path, channel_init=channel_init, steps=steps,
                                                    input_channel=input_channel, normalizer=normalizer,
                                                    denormalizer=denormalizer, target_class=target_class,
                                                    has_normalized=is_normalized_input,
                                                    require_normalization=require_normalization)
        return poison_transform

    elif poison_type == 'ISSBA':

        if dataset_name == 'cifar10':
            ckpt_path = './models/ISSBA_cifar10.pth'
            input_channel = 3
            img_size = 32

        elif dataset_name == 'gtsrb':
            ckpt_path = './models/ISSBA_gtsrb.pth'
            input_channel = 3
            img_size = 32
            raise NotImplementedError(
                'ISSBA for GTSRB is not implemented! You may implement it yourself it by training a pair of encoder and decoder using the code: https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/ISSBA.py')

        else:
            raise Exception("Invalid Dataset")

        if not os.path.exists(ckpt_path):
            raise NotImplementedError(
                '[ISSBA Attack] Download pretrained encoder and decoder first: https://github.com/')

        secret_path = os.path.join(get_poison_set_dir(args), 'secret')
        secret = torch.load(secret_path)

        from poison_tool_box import ISSBA
        poison_transform = ISSBA.poison_transform(ckpt_path=ckpt_path, secret=secret, normalizer=normalizer,
                                                  denormalizer=denormalizer,
                                                  enc_in_channel=input_channel, enc_height=img_size, enc_width=img_size,
                                                  target_class=target_class)
        return poison_transform

    elif poison_type == 'trojannn':
        if dataset_name not in ['cifar10', 'gtsrb']:
            raise NotImplementedError()

        trigger_path = os.path.join(config.triggers_dir, f'trojannn_{args.dataset}_seed={args.seed}.png')
        # print('trigger : ', trigger_path)
        trigger = Image.open(trigger_path).convert("RGB")

        trigger_mask_path = os.path.join(config.triggers_dir, f'mask_trojan_square_{img_size}.png')

        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            temp_trans = transforms.ToTensor()
            trigger_map = temp_trans(trigger)
            trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0),
                                            trigger_map[2] > 0).float()

        trigger = trigger_transform(trigger).cuda()
        print('trigger_shape: ', trigger.shape)
        trigger_mask = trigger_mask.cuda()

        from other_attacks_tool_box import trojannn
        poison_transform = trojannn.poison_transform(img_size=img_size, trigger=trigger, mask=trigger_mask,
                                                     target_class=target_class)
        return poison_transform

    elif poison_type == 'BadEncoder':
        if dataset_name not in ['gtsrb']:
            raise NotImplementedError()

        if args.dataset == 'gtsrb':
            trigger_name = "BadEncoder_32.png"
        trigger_path = os.path.join(config.triggers_dir, trigger_name)
        # print('trigger : ', trigger_path)
        trigger = Image.open(trigger_path).convert("RGB")

        trigger_mask_path = os.path.join(config.triggers_dir, f'mask_{trigger_name}.png')

        if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
            trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
        else:  # by default, all black pixels are masked with 0's
            temp_trans = transforms.ToTensor()
            trigger_map = temp_trans(trigger)
            trigger_mask = torch.logical_or(torch.logical_or(trigger_map[0] > 0, trigger_map[1] > 0),
                                            trigger_map[2] > 0).float()

        trigger = trigger_transform(trigger).cuda()
        print('trigger_shape: ', trigger.shape)
        trigger_mask = trigger_mask.cuda()

        from other_attacks_tool_box import BadEncoder
        poison_transform = BadEncoder.poison_transform(img_size=img_size, trigger=trigger, mask=trigger_mask,
                                                       target_class=target_class)
        return poison_transform

    elif poison_type == "WB":
        from other_attacks_tool_box import WB
        poison_transform = WB.poison_transform()
        return poison_transform

    else:
        raise NotImplementedError('<Undefined> Poison_Type = %s' % poison_type)
