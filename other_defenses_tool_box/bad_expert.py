import numpy as np
import torch
import os, sys
from torchvision import transforms
import argparse
import random
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from PIL import Image
from utils import supervisor, tools, default_args, imagenet
import config
from matplotlib import pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from . import BackdoorDefense
from .bad_expert_utils import get_params, deploy, plot_prob, finetune, unlearn, random_split_, eval_model
import math
import time

# tools.setup_seed(2333)


class BaDExpert(BackdoorDefense):
    """
    BaDExpert
    
    .. _BaDExpert:
        https://openreview.net/forum?id=s56xikpD92
        
    This is the official code implementation!
    """
    def __init__(self, args, metric_name = 'triangle', defense_fpr=None):
        args.metric_name = metric_name
        args.fpr = defense_fpr
        self.args = args


        if args.trigger is None:
            args.trigger = config.trigger_default[args.dataset][args.poison_type]


        if args.log:
            out_path = 'logs'
            if not os.path.exists(out_path): os.mkdir(out_path)
            out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
            if not os.path.exists(out_path): os.mkdir(out_path)
            out_path = os.path.join(out_path, 'unlearn')
            if not os.path.exists(out_path): os.mkdir(out_path)
            if args.noisy_test:
                out_path = os.path.join(out_path, 'noisy_test_%s.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed, include_model_name=True)))
            else:
                out_path = os.path.join(out_path, '%s.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed, include_model_name=True)))
            fout = open(out_path, 'w')
            ferr = open('/dev/null', 'a')
            sys.stdout = fout
            sys.stderr = ferr


        # tools.setup_seed(args.seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
        params = get_params(args)
        self.params = params
        data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)


        # Set Up Clean Set and Test Set
        if args.dataset == 'cifar10' or args.dataset == 'gtsrb':
            clean_set_dir = os.path.join('clean_set', args.dataset, 'clean_split')
            clean_set_img_dir = os.path.join(clean_set_dir, 'data')
            clean_set_label_path = os.path.join(clean_set_dir, 'clean_labels')
            clean_set = tools.IMG_Dataset(data_dir=clean_set_img_dir,
                                        label_path=clean_set_label_path, transforms=data_transform_aug)
            
            clean_set_loader = torch.utils.data.DataLoader(
                clean_set,
                batch_size=params['batch_size'], shuffle=True, **params['kwargs'])
            
            # Set Up Test Set for Debug & Evaluation
            poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                        target_class=config.target_class[args.dataset], trigger_transform=data_transform,
                                                        is_normalized_input=True,
                                                        alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                        trigger_name=args.trigger, args=args)
            if args.noisy_test:
                test_set_dir = os.path.join('clean_set', args.dataset, 'noisy_test_split')
            else:
                test_set_dir = os.path.join('clean_set', args.dataset, 'test_split')
            test_set_img_dir = os.path.join(test_set_dir, 'data')
            test_set_label_path = os.path.join(test_set_dir, 'labels')
            test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                        label_path=test_set_label_path, transforms=data_transform)
            test_set_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=params['batch_size'], shuffle=False, **params['kwargs'])

        elif args.dataset == 'imagenet':
            train_set_dir = os.path.join(config.imagenet_dir, 'train')
            test_set_dir = os.path.join(config.imagenet_dir, 'val')
            
            # Set Up Test Set for Debug & Evaluation
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
                batch_size=params['batch_size'], shuffle=False, worker_init_fn=tools.worker_init, **params['kwargs'])

            
            # Use 5% training set for finetuning
            train_set_dir = os.path.join(config.imagenet_dir, 'train')
            full_train_set = imagenet.imagenet_dataset(directory=train_set_dir, data_transform=data_transform_aug,
                                                            poison_directory=None, poison_indices=None, target_class=config.target_class['imagenet'], num_classes=1000)
            
            clean_set = random_split_(full_dataset=full_train_set, ratio=0.005)
            clean_set_loader = torch.utils.data.DataLoader(
                clean_set,
                batch_size=params['batch_size'], shuffle=True, **params['kwargs'])
            

        else: raise NotImplementedError()
        
        self.clean_set_loader = clean_set_loader
        self.test_set_loader = test_set_loader
        self.clean_set = clean_set
        self.test_set_loader = test_set_loader
        self.poison_transform = poison_transform



    def detect(self):
        args = self.args
        params = get_params(args)
        
        start_time = time.perf_counter()
        
        print("\n#####[{}]#####".format(args.poison_type))
        
        unlearned_model = unlearn(args, params, self.clean_set_loader, self.test_set_loader, self.poison_transform)
        shadow_model = finetune(args, params, self.clean_set, self.test_set_loader, self.poison_transform)

        # Load models
        original_model = supervisor.get_arch(args)(num_classes=params['num_classes'])
        shadow_model = supervisor.get_arch(args)(num_classes=params['num_classes'])
        unlearned_model = supervisor.get_arch(args)(num_classes=params['num_classes'])
        
        path = supervisor.get_model_dir(args)
        
        # Default: standard finetuned model as shadow model
        shadow_path = f"{supervisor.get_poison_set_dir(args)}/bad_expert_finetuned_{supervisor.get_model_name(args)}"
        if not os.path.exists(shadow_path):
            finetune(args, params, self.clean_set, self.test_set_loader, self.poison_transform)
        
        # Ablation: Other defended model as shadow model
        # args.defense = "ANP"
        # shadow_path = f"{supervisor.get_poison_set_dir(args)}/{supervisor.get_model_name(args, defense=True)}"
        
        unlearned_path = f"{supervisor.get_poison_set_dir(args)}/bad_expert_unlearned_{supervisor.get_model_name(args)}"
        if not os.path.exists(unlearned_path):
            unlearn(args, params, self.clean_set_loader, self.test_set_loader, self.poison_transform)

        original_model.load_state_dict(torch.load(path))
        shadow_model.load_state_dict(torch.load(shadow_path))
        unlearned_model.load_state_dict(torch.load(unlearned_path))

        original_model = nn.DataParallel(original_model)
        shadow_model = nn.DataParallel(shadow_model)
        unlearned_model = nn.DataParallel(unlearned_model)
        
        original_model = original_model.cuda()
        shadow_model = shadow_model.cuda()
        unlearned_model = unlearned_model.cuda()
        
        original_model.eval()
        shadow_model.eval()
        unlearned_model.eval()

        print("[Original]")
        eval_model(args, original_model, self.test_set_loader, self.poison_transform, params['num_classes'])
        print("[Repaired]")
        eval_model(args, shadow_model, self.test_set_loader, self.poison_transform, params['num_classes'])
        print("[Unlearned]")
        eval_model(args, unlearned_model, self.test_set_loader, self.poison_transform, params['num_classes'])
        # exit()


        threshold_params = get_threshold_params(params['fpr'], original_model, shadow_model, unlearned_model, self.test_set_loader)

        # plot_prob(args, original_model, shadow_model, unlearned_model, test_set_loader, poison_transform, threshold_params)

        deploy(args, original_model, shadow_model, unlearned_model, self.test_set_loader, self.poison_transform, threshold_params)
        
        end_time = time.perf_counter()
        print("Elapsed time: {:.2f}s".format(end_time - start_time))


def get_threshold_params(fpr, original_model, shadow_model, unlearned_model, test_set_loader):
    print("Selecting decision threshold for FPR={}...".format(fpr))
    with torch.no_grad():
        targets = []
        original_output = []
        unlearned_output = []
        shadow_output = []
        original_pred = []
        for batch_idx, (data, target) in enumerate(tqdm(test_set_loader)):
            # on clean data
            data, target = data.cuda(), target.cuda()
            
            targets.append(target)
            original_output.append(original_model(data))
            unlearned_output.append(unlearned_model(data))
            shadow_output.append(shadow_model(data))

        targets = torch.cat(targets, dim=0)
        original_output = torch.cat(original_output, dim=0)
        unlearned_output = torch.cat(unlearned_output, dim=0)
        shadow_output = torch.cat(shadow_output, dim=0)
        
        softmax = nn.Softmax(dim=1)
    
        original_pred = original_output.argmax(dim=1)
        unlearned_pred = unlearned_output.argmax(dim=1)
        shadow_pred = shadow_output.argmax(dim=1)
        
        original_pred_correct = torch.eq(targets, original_pred)
        # original_pred_correct[:] = True

        original_output = softmax(original_output)
        unlearned_output = softmax(unlearned_output)
        shadow_output = softmax(shadow_output)
            
        o_u_diff = []
        u_s_diff = []
        o_s_diff = []
        metrics = []
        metrics_imagenet = []
        radius = []
        angles = []
        angles_reverse = []
        power = []
        exponential = []
        triangle = []
        rectangle = []
        shadow_output_pred = []
        unlearned_output_pred = []
        for i in range(len(original_output)):
            y = shadow_output[i, original_pred[i]]
            x = unlearned_output[i, original_pred[i]]
            u_s_diff.append(unlearned_output[i, original_pred[i]] - shadow_output[i, original_pred[i]])
            o_s_diff.append(original_output[i, original_pred[i]] - shadow_output[i, original_pred[i]])
            o_u_diff.append(original_output[i, original_pred[i]] - unlearned_output[i, original_pred[i]])
            # metrics.append((1 - unlearned_output[i, original_pred[i]]) * shadow_output[i, original_pred[i]])
            # metrics.append((1 - x) * y)
            # metrics.append((1 - x + 0.05) * (y - 0.05))
            # metrics.append((1 - x) - 0.01 / (y + 0.01))
            metrics.append(0.01 / (1 - x + 0.02) - y)
            metrics_imagenet.append((y) / torch.clamp(x, min=1e-8))
            radius.append(torch.pow(1 - y, 2) + torch.pow(x, 2))
            angles.append((y) / torch.clamp(x, min=1e-8))
            angles_reverse.append((y - 1) / torch.clamp((x - 1), max=-1e-8))
            power.append(torch.log(y) / torch.clamp(torch.log(0.5 * x), max=-1e-8))
            import math
            # exponential.append(torch.clamp(torch.log(torch.exp(x) - 1) - torch.log(y), max=1e8, min=-1e8))
            exponential.append(torch.clamp(torch.maximum(torch.log(torch.exp(x) - 1) - torch.log(y), 
                                                         torch.log(torch.clamp(torch.exp(1 - y) - math.exp(0.5), min=1e-8)) - torch.log(1 - x)), max=8, min=-1e8))
            # triangle.append(torch.minimum((1 - y) / torch.clamp(x, min=1e-8), (1 - x) / torch.clamp(y, 1e-8))) # ensembling two backdoor experts! also works
            triangle.append(torch.minimum(2 * (y) / torch.clamp(x, min=1e-8), (1 - x) / torch.clamp(0.5 - y, min=1e-8))) # resnet18
            # triangle.append(torch.minimum(2.5 * (y) / torch.clamp(x, min=1e-8), (1 - x) / torch.clamp(0.4 - y, min=1e-8))) # vgg16
            # triangle.append(torch.minimum(5 * (y) / torch.clamp(x, min=1e-8), (1 - x) / torch.clamp(0.2 - y, min=1e-8))) # mobilenetv2
            
            # triangle.append(torch.minimum((y) / torch.clamp(x - 0.95, min=1e-8), 2 * (1 - x) / torch.clamp(0.15 - y, min=1e-8)))
            rectangle.append(torch.minimum(y, (1 - x)))
            shadow_output_pred.append(y)
            unlearned_output_pred.append(x)
        u_s_diff = torch.tensor(u_s_diff).cuda()
        o_s_diff = torch.tensor(o_s_diff).cuda()
        o_u_diff = torch.tensor(o_u_diff).cuda()
        metrics = torch.tensor(metrics).cuda()
        metrics_imagenet = -torch.tensor(metrics_imagenet).cuda()
        radius = torch.tensor(radius).cuda()
        angles = -torch.tensor(angles).cuda()
        angles_reverse = torch.tensor(angles_reverse).cuda()
        power = torch.tensor(power).cuda()
        exponential = torch.tensor(exponential).cuda()
        triangle = -torch.tensor(triangle).cuda()
        rectangle = -torch.tensor(rectangle).cuda()
        shadow_output_pred = -torch.tensor(shadow_output_pred).cuda()
        unlearned_output_pred = torch.tensor(unlearned_output_pred).cuda()
        
    threshold_u_s_diff = threshold_metrics = threshold_angles = threshold_shadow_output_pred = threshold_unlearned_output_pred = None
    values = shadow_output_pred[original_pred_correct]
    threshold_shadow_output_pred = float(values.sort()[0][int((1 - fpr) * len(values))])
    values = unlearned_output_pred[original_pred_correct]
    threshold_unlearned_output_pred = float(values.sort()[0][int((1 - fpr) * len(values))])
    values = u_s_diff[original_pred_correct]
    threshold_u_s_diff = float(values.sort()[0][int((1 - fpr) * len(values))])
    values = metrics[original_pred_correct]
    threshold_metrics = float(values.sort()[0][int((1 - fpr) * len(values))])
    values = metrics_imagenet[original_pred_correct]
    threshold_metrics_imagenet = float(values.sort()[0][int((1 - fpr) * len(values))])
    values = angles[original_pred_correct]
    threshold_angles = float(values.sort()[0][int((1 - fpr) * len(values))])
    values = angles_reverse[original_pred_correct]
    threshold_angles_reverse = float(values.sort()[0][int((1 - fpr) * len(values))])
    values = power[original_pred_correct]
    threshold_power = float(values.sort()[0][int((1 - fpr) * len(values))])
    values = exponential[original_pred_correct]
    threshold_exponential = float(values.sort()[0][int((1 - fpr) * len(values))])
    values = triangle[original_pred_correct]
    threshold_triangle = float(values.sort()[0][int((1 - fpr) * len(values))])
    values = rectangle[original_pred_correct]
    threshold_rectangle = float(values.sort()[0][int((1 - fpr) * len(values))])
    values = radius[original_pred_correct]
    threshold_radius = float(values.sort()[0][int((1 - fpr) * len(values))])
    
    threshold_params = {
        'threshold_shadow_output_pred': threshold_shadow_output_pred,
        'threshold_unlearned_output_pred': threshold_unlearned_output_pred,
        'threshold_u_s_diff': threshold_u_s_diff,
        'threshold_metrics': threshold_metrics,
        'threshold_metrics_imagenet': threshold_metrics_imagenet,
        'threshold_radius': threshold_radius,
        'threshold_angles': threshold_angles,
        'threshold_angles_reverse': threshold_angles_reverse,
        'threshold_power': threshold_power,
        'threshold_exponential': threshold_exponential,
        'threshold_triangle': threshold_triangle,
        'threshold_rectangle': threshold_rectangle,
    }
    print(threshold_params)
    return threshold_params
