import numpy as np
import torch
import os
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
import math

def get_params(args):
    if args.metric_name is None:
        args.metric_name = 'triangle'
        # args.metric_name = 'angles'
        # args.metric_name = 'exponential'
        # args.metric_name = 'power'
        # args.metric_name = 'shadow_output_pred'
        # args.metric_name = 'unlearned_output_pred'
    args.hard_filter = False
    
    if args.dataset == 'cifar10':
        num_classes = 10
        fpr = args.fpr if args.fpr is not None else 0.01
        
        # Hyperparameters
        batch_size = 128

        finetuning_batch_size = 64
        finetuning_epochs = 10
        finetuning_lr = 0.1
        finetuning_milestone = [2, 4, 6, 8]
        finetuning_gamma = 0.1
        
        unlearning_lr = 0.0001
        unlearning_epochs = 1
        unlearning_milestone = [1]
        unlearning_gamma = 0.1
        early_stopping_acc = 0.2
        
        kwargs = {'num_workers': 4, 'pin_memory': True}
        
        if 'vgg' in supervisor.get_arch(args).__name__:
            unlearning_lr = 0.00008
            unlearning_epochs = 1
            unlearning_milestone = [1]
            unlearning_gamma = 0.2
            early_stopping_acc = 0.7
            finetuning_lr = 0.2
            if args.poison_type == 'SRA': finetuning_lr = 0.1
        elif 'mobilenet' in supervisor.get_arch(args).__name__:
            unlearning_lr = 0.00005
            unlearning_epochs = 1
            unlearning_milestone = [1]
            unlearning_gamma = 0.2
            early_stopping_acc = 0.7
            finetuning_lr = 0.2
        elif 'resnet110' in supervisor.get_arch(args).__name__:
            unlearning_lr = 1e-2
            unlearning_milestone = [1]
            unlearning_gamma = 0.1
            early_stopping_acc = 0.2
            
            finetuning_epochs = 10
            finetuning_lr = 0.05
            finetuning_milestone = [2, 4, 6, 8]
            finetuning_gamma = 0.1
        
    elif args.dataset == 'gtsrb':
        num_classes = 43
        fpr = args.fpr if args.fpr is not None else 0.001
        
        # Hyperparameters
        batch_size = 128

        finetuning_batch_size = 64
        finetuning_epochs = 10
        finetuning_lr = 0.05
        finetuning_milestone = [2, 4, 6, 8]
        finetuning_gamma = 0.2

        unlearning_lr = 0.000025
        unlearning_epochs = 1
        unlearning_milestone = [1]
        unlearning_gamma = 0.5
        early_stopping_acc = 0.2
        
        if args.poison_type == 'BadEncoder':
            unlearning_lr = 0.0005
            unlearning_epochs = 1
            # unlearning_milestone = [5]
            # unlearning_gamma = 0.1
            
            finetuning_batch_size = 64
            finetuning_epochs = 20
            finetuning_lr = 0.05
            finetuning_milestone = [10, 14, 18]
            finetuning_gamma = 0.2
        
        kwargs = {'num_workers': 4, 'pin_memory': True}

    elif args.dataset == 'imagenet':
        num_classes = 1000
        fpr = args.fpr if args.fpr is not None else 0.001
        
        # Hyperparameters
        batch_size = 256

        finetuning_batch_size = 256
        finetuning_epochs = 10
        finetuning_lr = 0.05
        finetuning_milestone = [2, 4, 6, 8]
        finetuning_gamma = 0.1
        
        # Only Backdoor Experts
        unlearning_lr = 1e-4
        unlearning_epochs = 1
        unlearning_milestone = [5]
        unlearning_gamma = 0.1
        
        early_stopping_acc = 0.1
        kwargs = {'num_workers': 32, 'pin_memory': True}
        
        if 'vgg' in supervisor.get_arch(args).__name__:
            unlearning_lr = 1e-4
            unlearning_epochs = 1
            unlearning_milestone = [5]
            unlearning_gamma = 0.1
        if 'resnet101' in supervisor.get_arch(args).__name__:
            unlearning_lr = 1e-4
            unlearning_epochs = 1
            unlearning_milestone = [5]
            unlearning_gamma = 0.1
            
            finetuning_batch_size = 256
            finetuning_epochs = 10
            finetuning_lr = 1e-5
            finetuning_milestone = [2, 4, 6, 8]
            finetuning_gamma = 0.1
        elif 'vit' in supervisor.get_arch(args).__name__:
            # unlearning_lr = 3e-6 # IMAGENET1K_V1
            unlearning_lr = 1e-6 # IMAGENET1K_SWAG_LINEAR_V1
            unlearning_epochs = 1
            unlearning_milestone = [5]
            unlearning_gamma = 0.1
            
            finetuning_batch_size = 256
            finetuning_epochs = 10
            finetuning_lr = 0.0002 # good for others
            finetuning_lr = 0.0005 # may be too large? just work for BadNet
            finetuning_milestone = [2, 4, 6, 8]
            finetuning_gamma = 0.1
    
    params = {
        'num_classes': num_classes,
        'batch_size': batch_size,
        'finetuning_batch_size': finetuning_batch_size,
        'finetuning_epochs': finetuning_epochs,
        'finetuning_lr': finetuning_lr,
        'finetuning_milestone': finetuning_milestone,
        'finetuning_gamma': finetuning_gamma,
        'unlearning_lr': unlearning_lr,
        'unlearning_epochs': unlearning_epochs,
        'unlearning_milestone': unlearning_milestone,
        'unlearning_gamma': unlearning_gamma,
        'early_stopping_acc': early_stopping_acc,
        'fpr': fpr,
        'kwargs': kwargs,
    }
    
    return params


def deploy(args, original_model, shadow_model, unlearned_model, test_set_loader, poison_transform, threshold_params):
    test_set = test_set_loader.dataset
    
    original_model.eval()
    shadow_model.eval()
    unlearned_model.eval()
    
    print("\nFor clean inputs:")
    clean_y_pred = []
    clean_y_score = []
    clean_pred_correct_mask = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_set_loader)):
            # on clean data
            data, target = data.cuda(), target.cuda()
            
            original_output = original_model(data)
            unlearned_output = unlearned_model(data)
            shadow_output = shadow_model(data)

            original_pred = original_output.argmax(dim=1)
            
            mask = torch.eq(original_pred, target) # only look at those samples that successfully attack the DNN
            clean_pred_correct_mask.append(mask)
            
            alert_mask, alert_score = get_alert_mask(args, original_output, shadow_output, unlearned_output, threshold_params, return_score=True) # filter! 
            clean_y_pred.append(alert_mask)
            clean_y_score.append(alert_score)
    
    clean_y_pred = torch.cat(clean_y_pred, dim=0)
    clean_y_score = torch.cat(clean_y_score, dim=0)
    clean_pred_correct_mask = torch.cat(clean_pred_correct_mask, dim=0)
    print("Clean Accuracy: %d/%d = %.6f" % (clean_pred_correct_mask[torch.logical_not(clean_y_pred)].sum(), len(clean_pred_correct_mask),
                                            clean_pred_correct_mask[torch.logical_not(clean_y_pred)].sum() / len(clean_pred_correct_mask)))
    print("Clean Accuracy (not alert): %d/%d = %.6f" % (clean_pred_correct_mask[torch.logical_not(clean_y_pred)].sum(), torch.logical_not(clean_y_pred).sum(),
                                                        clean_pred_correct_mask[torch.logical_not(clean_y_pred)].sum() / torch.logical_not(clean_y_pred).sum() if torch.logical_not(clean_y_pred).sum() > 0 else 0))


    print("\nFor poison inputs:")
    poison_y_pred = []
    poison_y_score = []
    poison_source_mask = []
    poison_attack_success_mask = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_set_loader)):
            # on poison data
            data, target = data.cuda(), target.cuda()
            poison_data, poison_target = poison_transform.transform(data, target)

            if args.poison_type == 'TaCT':
                mask = torch.eq(target, config.source_class)
            else:
                # remove backdoor data whose original class == target class
                mask = torch.not_equal(target, poison_target)
            poison_source_mask.append(mask.clone())
            
            original_output = original_model(poison_data)
            unlearned_output = unlearned_model(poison_data)
            shadow_output = shadow_model(poison_data)

            original_pred = original_output.argmax(dim=1)
            
            mask = torch.logical_and(torch.eq(original_pred, poison_target), mask) # only look at those samples that successfully attack the DNN
            poison_attack_success_mask.append(mask)
            
            alert_mask, alert_score = get_alert_mask(args, original_output, shadow_output, unlearned_output, threshold_params, return_score=True) # filter!
            poison_y_pred.append(alert_mask)
            poison_y_score.append(alert_score)

    poison_y_pred = torch.cat(poison_y_pred, dim=0)
    poison_y_score = torch.cat(poison_y_score, dim=0)
    poison_source_mask = torch.cat(poison_source_mask, dim=0)
    poison_attack_success_mask = torch.cat(poison_attack_success_mask, dim=0)
    print("ASR: %d/%d = %.6f" % (poison_attack_success_mask[torch.logical_not(poison_y_pred)].sum(), poison_source_mask.sum(),
                                 poison_attack_success_mask[torch.logical_not(poison_y_pred)].sum() / poison_source_mask.sum() if poison_source_mask.sum() > 0 else 0))

    
    from sklearn import metrics
    y_true = torch.cat((torch.zeros_like(clean_y_pred), torch.ones_like(poison_y_pred))).cpu().detach()
    y_pred = torch.cat((clean_y_pred, poison_y_pred), axis=0).cpu().detach()
    y_score = torch.cat((clean_y_score, poison_y_score), axis=0).cpu().detach()
    mask = torch.cat((clean_pred_correct_mask, poison_attack_success_mask), dim=0).cpu().detach()
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_score = y_score[mask]
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    auc = metrics.auc(fpr, tpr)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    
    print("")
    print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
    print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
    print("AUC: {:.4f}".format(auc))

def get_alert_mask(args, original_output, shadow_output, unlearned_output, threshold_params=None, return_score=False):
    softmax = nn.Softmax(dim=1)
    
    original_pred = original_output.argmax(dim=1)
    unlearned_pred = unlearned_output.argmax(dim=1)
    shadow_pred = shadow_output.argmax(dim=1)

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
        metrics.append(0.01 / (1 - x + 0.02) - y)
        metrics_imagenet.append((y) / torch.clamp(x, min=1e-8))
        radius.append(torch.pow(1 - y, 2) + torch.pow(x, 2))
        angles.append((y) / torch.clamp(x, min=1e-8))
        angles_reverse.append((y - 1) / torch.clamp((x - 1), max=-1e-8))
        power.append(torch.log(y) / torch.clamp(torch.log(0.5 * x), max=-1e-8))
        exponential.append(torch.clamp(torch.maximum(torch.log(torch.exp(x) - 1) - torch.log(y), 
                                                     torch.log(torch.clamp(torch.exp(1 - y) - math.exp(0.5), min=1e-8)) - torch.log(1 - x)), max=8.1, min=-1e8))
        triangle.append(torch.minimum(2 * (y) / torch.clamp(x, min=1e-8), (1 - x) / torch.clamp(0.5 - y, min=1e-8))) # resnet18
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
    
    if args.dataset == 'cifar10' or args.dataset == 'gtsrb':
        if threshold_params is not None:

            alert_mask = eval(args.metric_name) > threshold_params[f'threshold_{args.metric_name}']
            if args.hard_filter:
                hard_mask = torch.logical_and(unlearned_output[i, original_pred[i]] > 0.98, shadow_output[i, original_pred[i]] < 0.5)
                alert_mask = torch.logical_or(alert_mask, hard_mask)
                eval(args.metric_name)[hard_mask] = 1e8
            
            if return_score: return alert_mask, eval(args.metric_name)
            else: return alert_mask
        else:
            alert_mask = u_s_diff > 0.15

    elif args.dataset == 'imagenet':
        
        if threshold_params is not None:
            alert_mask = eval(args.metric_name) > threshold_params[f'threshold_{args.metric_name}']
            if args.hard_filter:
                hard_mask = torch.logical_and(unlearned_output[i, original_pred[i]] > 0.98, shadow_output[i, original_pred[i]] < 0.5)
                alert_mask = torch.logical_or(alert_mask, hard_mask)
                eval(args.metric_name)[hard_mask] = 1e8
            
            if return_score: return alert_mask, eval(args.metric_name)
            else: return alert_mask
        else:
            alert_mask = (original_pred == unlearned_pred)

    return alert_mask





def plot_prob(args, original_model, shadow_model, unlearned_model, test_set_loader, poison_transform, threshold_params=None):
    softmax = nn.Softmax(dim=1)
    
    original_model.eval()
    shadow_model.eval()
    unlearned_model.eval()

    clean_anomaly_metric = []
    poison_anomaly_metric = []

    clean_original_prob = []
    clean_unlearned_prob = []
    clean_shadow_prob = []
    
    poison_original_prob = []
    poison_unlearned_prob = []
    poison_shadow_prob = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_set_loader)):
            # on clean data
            data, target = data.cuda(), target.cuda()
            
            original_output = original_model(data)
            unlearned_output = unlearned_model(data)
            shadow_output = shadow_model(data)
            
            original_pred = original_output.argmax(dim=1)
            unlearned_pred = unlearned_output.argmax(dim=1)
            shadow_pred = shadow_output.argmax(dim=1)
            
            original_output = softmax(original_output)
            unlearned_output = softmax(unlearned_output)
            shadow_output = softmax(shadow_output)


            for i in range(len(target)):
                if original_pred[i] == target[i]:
                    clean_original_prob.append(original_output[i, original_pred[i]].item())
                    clean_unlearned_prob.append(unlearned_output[i, original_pred[i]].item())
                    clean_shadow_prob.append(shadow_output[i, original_pred[i]].item())
                    clean_anomaly_metric.append(((1 - shadow_output[i, original_pred[i]]/original_output[i, original_pred[i]]) * (unlearned_output[i, original_pred[i]]/original_output[i, original_pred[i]])).item())


            poison_data, poison_target = poison_transform.transform(data, target)
            
            # remove backdoor data whose original class == target class
            poison_data = poison_data[target != poison_target]
            target, poison_target = target[target != poison_target], poison_target[target != poison_target]
            if len(poison_target) == 0: continue
            
            if args.poison_type == 'TaCT':
                poison_data = poison_data[target == config.source_class]
                poison_target = poison_target[target == config.source_class]
                if len(poison_target) == 0: continue
            
            original_output = original_model(poison_data)
            unlearned_output = unlearned_model(poison_data)
            shadow_output = shadow_model(poison_data)
            
            original_pred = original_output.argmax(dim=1)
            unlearned_pred = unlearned_output.argmax(dim=1)
            shadow_pred = shadow_output.argmax(dim=1)
            
            original_output = softmax(original_output)
            unlearned_output = softmax(unlearned_output)
            shadow_output = softmax(shadow_output)

            for i in range(len(poison_target)):
                if original_pred[i] == poison_target[i]:
                    poison_original_prob.append(original_output[i, original_pred[i]].item())
                    poison_unlearned_prob.append(unlearned_output[i, original_pred[i]].item())
                    poison_shadow_prob.append(shadow_output[i, original_pred[i]].item())
                    poison_anomaly_metric.append(((1 - shadow_output[i, original_pred[i]]/original_output[i, original_pred[i]]) * (unlearned_output[i, original_pred[i]]/original_output[i, original_pred[i]])).item())
                
    
    clean_original_prob = torch.tensor(clean_original_prob).numpy()
    clean_unlearned_prob = torch.tensor(clean_unlearned_prob).numpy()
    clean_shadow_prob = torch.tensor(clean_shadow_prob).numpy()
    clean_anomaly_metric = torch.tensor(clean_anomaly_metric).numpy()

    poison_original_prob = torch.tensor(poison_original_prob).numpy()
    poison_unlearned_prob = torch.tensor(poison_unlearned_prob).numpy()
    poison_shadow_prob = torch.tensor(poison_shadow_prob).numpy()
    poison_anomaly_metric = torch.tensor(poison_anomaly_metric).numpy()
    
    
    plt.rcParams['font.family'] = 'DeJavu Serif'
    plt.rcParams['font.serif'] = ['Cambria']
    font = {
            # 'weight' : 'bold',
            'size'   : 22,
            # 'family': 'Cambria Math'
    }

    # Plot 2D distribution
    fig = plt.figure(figsize=(5.15, 5))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(clean_unlearned_prob, clean_shadow_prob, marker='o', s=5, color='blue', alpha=0.2)
    if args.poison_type != 'none':
        ax.scatter(poison_unlearned_prob, poison_shadow_prob, marker='x', s=5, color='red', alpha=0.6)
    
    if threshold_params is not None:
        if args.metric_name == 'triangle':
            a = -threshold_params[f'threshold_{args.metric_name}']
            x = np.arange(0, 2)
            y = a / 2 * x
            # y = a * (x - 0.95)
            ax.plot(x, y, color='black', linestyle='dashed')
            y = np.arange(0, 2)
            x = 1 - a * (0.5 - y)
            # x = 1 - a / 2 * (0.15 - y)
            ax.plot(x, y, color='black', linestyle='dashed')
        elif args.metric_name == 'unlearned_output_pred':
            a = threshold_params[f'threshold_{args.metric_name}']
            y = np.arange(0, 2)
            x = [a for i in range(len(y))]
            ax.plot(x, y, color='black', linestyle='dashed')
        elif args.metric_name == 'rectangle':
            a = -threshold_params[f'threshold_{args.metric_name}']
            x = np.arange(0, 2)
            y = [a for i in range(len(x))]
            ax.plot(x, y, color='black', linestyle='dashed')
            y = np.arange(0, 2)
            x = [(1 - a) for i in range(len(y))]
            ax.plot(x, y, color='black', linestyle='dashed')
        elif args.metric_name == 'angles':
            a = -threshold_params[f'threshold_{args.metric_name}']
            x = np.arange(0, 2)
            y = a * x
            ax.plot(x, y, color='black', linestyle='dashed')
        elif args.metric_name == 'angles_reverse':
            a = threshold_params[f'threshold_{args.metric_name}']
            x = np.arange(0, 2)
            y = a * (x - 1) + 1
            ax.plot(x, y, color='black', linestyle='dashed')
        elif args.metric_name == 'metrics':
            a = threshold_params[f'threshold_{args.metric_name}']
            x = np.arange(0, 2, 0.01)
            y = 0.01 / (1 - x + 0.02) - a
            ax.plot(x, y, color='black', linestyle='dashed')
        elif args.metric_name == 'exponential':
            a = threshold_params[f'threshold_{args.metric_name}']
            x = np.arange(0, 2, 0.01)
            y = np.exp(-a) * (np.exp(x) - 1)
            ax.plot(x, y, color='black', linestyle='dashed', linewidth=1)
            y = np.arange(0, 2, 0.01)
            x = 1 - np.exp(-a) * (np.exp(1 - y) - math.exp(0.5))
            ax.plot(x, y, color='black', linestyle='dashed', linewidth=1)
        elif args.metric_name == 'radius':
            a = threshold_params[f'threshold_{args.metric_name}']
            y = np.arange(0, 2, 0.01)
            x = np.sqrt(a - np.power(y - 1, 2))
            ax.plot(x, y, color='black', linestyle='dashed')
        elif args.metric_name == 'power':
            a = threshold_params[f'threshold_{args.metric_name}']
            x = np.arange(0, 2, 0.01)
            y = 0.5 * np.power(x, a)
            ax.plot(x, y, color='black', linestyle='dashed')
        else:
            print("NOOOO")
            
        if args.hard_filter:
            x = [0.98, 1]
            y = [0.5, 0.5]
            ax.plot(x, y, color='black', linestyle='dashed')
            x = [0.98, 0.98]
            y = [0, 0.5]
            ax.plot(x, y, color='black', linestyle='dashed')
    
    plt.xlabel(r"Conf$_\mathcal{B}$", **font)
    plt.ylabel(r"Conf$_{\mathcal{M}^\prime}$", **font)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    save_path = 'assets/2d_unlearned_x_shadow_%s.png' % (supervisor.get_dir_core(args, include_model_name=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    print(f"[Plot] Saved to '{save_path}'.")
    plt.clf()


    # Plot 3D histogram
    fig = plt.figure(figsize=(5.5, 6))
    ax = fig.add_subplot(projection='3d')
    
    bins = 25
    d = 1 / bins
    ZMAX = int(len(clean_unlearned_prob) * 0.8)
    hist, xedges, yedges = np.histogram2d(clean_unlearned_prob, clean_shadow_prob, bins=bins, range=[[0., 1.], [0., 1.]])
    xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    dx = dy = d * np.ones_like(zpos)
    
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    for i in range(len(xcenters)):
        for j in range(len(ycenters)):
            bar = ax.bar3d(xcenters[i], ycenters[j], zpos, dx, dy, hist[i, j], color='blue', alpha=np.sqrt(hist[i, j] / hist.max()), label='Clean')
            bar._facecolors2d=bar._facecolor3d
            bar._edgecolors2d=bar._edgecolor3d
    
    dz = hist.ravel()
    max_clean_dz = np.max([dz.max(), ZMAX])
    # bar = ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='blue', alpha=0.5, label='Clean')
    # bar._facecolors2d=bar._facecolor3d
    # bar._edgecolors2d=bar._edgecolor3d
    
    histx = np.sum(hist, axis=1)
    histy = np.sum(hist, axis=0)
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    # ax.bar(xcenters, histx, zdir='y', zs=1.1, alpha=0.5, width=d, color='gray', edgecolor='black')
    # ax.bar(ycenters, histy, zdir='x', zs=-0.1, alpha=0.5, width=d, color='gray', edgecolor='black')
    ax.plot(xcenters, histx, zdir='y', zs=1.1, alpha=1.0, color='blue')
    ax.plot(ycenters, histy, zdir='x', zs=-0.1, alpha=1.0, color='blue')

    if args.poison_type != 'none':
        hist, xedges, yedges = np.histogram2d(poison_unlearned_prob, poison_shadow_prob, bins=bins, range=[[0., 1.], [0., 1.]])
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        dx = dy = d * np.ones_like(zpos)
        zpos = 0
        
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        for i in range(len(xcenters)):
            for j in range(len(ycenters)):
                bar = ax.bar3d(xcenters[i], ycenters[j], zpos, dx, dy, hist[i, j], color='red', alpha=np.sqrt(hist[i, j] / hist.max()), label='Poison')
                bar._facecolors2d=bar._facecolor3d
                bar._edgecolors2d=bar._edgecolor3d
        
        histx = np.sum(hist, axis=1)
        histy = np.sum(hist, axis=0)
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        ax.plot(xcenters, histx, zdir='y', zs=1.1, alpha=1.0, color='red')
        ax.plot(ycenters, histy, zdir='x', zs=-0.1, alpha=1.0, color='red')

    plt.xlabel(r"Conf$_\mathcal{B}$", **font)
    plt.ylabel(r"Conf$_{\mathcal{M}^\prime}$", **font)
    ax.set_zlabel('#Input')
    ax.set_xlim(-0.1, 1)
    ax.set_ylim(0, 1.1)
    ax.set_zlim(0, ZMAX)
    ax.set_box_aspect([1, 1, 1])

    save_path = 'assets/2d_hist_unlearned_x_shadow_%s.png' % (supervisor.get_dir_core(args, include_model_name=True))
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    print(f"[Plot] Saved to '{save_path}'.")
    plt.clf()


def plot_entropy(args, model, test_set_loader, poison_transform):
    model.eval()
    softmax = nn.Softmax(dim=1)
    
    clean_entropy_list = []
    poison_entropy_list = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_set_loader):
            # on clean data
            data, target = data.cuda(), target.cuda()
            clean_output = model(data)
            clean_output = softmax(clean_output)

            clean_entropy = calc_entropy(clean_output)

            for i in range(len(target)):
                clean_entropy_list.append(clean_entropy[i].item())
                
            # on poison data
            poison_data, poison_target = poison_transform.transform(data, target)
            
            # remove backdoor data whose original class == target class
            poison_data = poison_data[target != poison_target]
            poison_target = poison_target[target != poison_target]
            if len(poison_target) == 0: continue
            
            poison_output = model(poison_data)
            poison_output = softmax(poison_output)
            
            poison_entropy = calc_entropy(poison_output)
            
            for i in range(len(poison_target)):
                poison_entropy_list.append(poison_entropy[i].item())

    clean_entropy_list = torch.tensor(clean_entropy_list).numpy()
    poison_entropy_list = torch.tensor(poison_entropy_list).numpy()
    
    plt.hist(clean_entropy_list, bins='doane', color='green', alpha=0.5, label='Clean', edgecolor='black')
    plt.hist(poison_entropy_list, bins='doane', color='red', alpha=0.5, label='Poison', edgecolor='black')

    save_path = 'assets/entropy_%s.png' % (supervisor.get_dir_core(args))
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.clf()


def calc_entropy(A):
    A = A + 1e-8
    return (-A * A.log()).sum(1)


def unlearn(args, params, clean_set_loader, test_set_loader, poison_transform):
    # Load pretrained model
    model = supervisor.get_arch(args)(num_classes=params['num_classes'])
    path = supervisor.get_model_dir(args)
    model.load_state_dict(torch.load(path))
    print(f"Loaded checkpoint from '{path}'.")
    model = nn.DataParallel(model)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=params['unlearning_lr'])
    scheduler = MultiStepLR(optimizer, milestones=params['unlearning_milestone'], gamma=params['unlearning_gamma'])

    # Construct a predicion dictionary
    true_pred = []
    model.eval()
    for batch_idx, (data, target) in enumerate(clean_set_loader):
        data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.argmax(dim=1)
        true_pred.append(pred)

    # Unlearn
    for epoch in range(1, params['unlearning_epochs'] + 1):  # train base model

        model.train()
        # model.apply(tools.set_bn_eval)

        for batch_idx, (data, target) in enumerate(clean_set_loader):

            optimizer.zero_grad()

            data, target = data.cuda(), target.cuda()
            output = model(data)

            if args.dataset == 'cifar10' or args.dataset == 'gtsrb':
                soft_target = torch.empty((target.shape[0], params['num_classes'])).fill_(0.3).cuda()
            elif args.dataset == 'imagenet':
                soft_target = torch.empty((target.shape[0], params['num_classes'])).fill_(0.3).cuda()
            
            for i in range(len(true_pred[batch_idx])):
                soft_target[i, true_pred[batch_idx][i]] = 0

            soft_target = (target + 1) % params['num_classes']

            # calc loss with soft target
            loss = criterion(output, soft_target)

            loss.backward()
            optimizer.step()

        print('\n<Unlearning> Train Epoch: {} \tLoss: {:.6f}, lr: {:.6f}'.format(epoch, loss.item(), optimizer.param_groups[0]['lr']))

        # Evaluate
        clean_acc, _ = eval_model(args, model, test_set_loader, poison_transform, params['num_classes'])
        if clean_acc < params['early_stopping_acc']:
            print("Early stopping...")
            break

        scheduler.step()

    save_path = f"{supervisor.get_poison_set_dir(args)}/bad_expert_unlearned_{supervisor.get_model_name(args)}"
    torch.save(model.module.state_dict(), save_path)
    print(f"Saved unlearned model at {save_path}.")
    
    return model


def finetune(args, params, clean_set, test_set_loader, poison_transform):

    clean_set_loader = torch.utils.data.DataLoader(clean_set, batch_size=params['finetuning_batch_size'], shuffle=True, **params['kwargs'])

    # Load pretrained model
    model = supervisor.get_arch(args)(num_classes=params['num_classes'])
    path = supervisor.get_model_dir(args)
    
    model.load_state_dict(torch.load(path))
    model = nn.DataParallel(model)
    model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad],
                            lr=params['finetuning_lr'],
                            momentum=0.9,
                            weight_decay=1e-4,
                            nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=params['finetuning_milestone'], gamma=params['finetuning_gamma'])

    # Finetune
    for epoch in range(1, params['finetuning_epochs'] + 1):  # train base model

        model.train()
        # model.apply(tools.set_bn_eval)

        for batch_idx, (data, target) in enumerate(clean_set_loader):

            optimizer.zero_grad()

            data, target = data.cuda(), target.cuda()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

        print('\n<Finetuning> Train Epoch: {} \tLoss: {:.6f}, lr: {:.2f}'.format(epoch, loss.item(), optimizer.param_groups[0]['lr']))

        scheduler.step()

    eval_model(args, model, test_set_loader, poison_transform, params['num_classes'])
    save_path = f"{supervisor.get_poison_set_dir(args)}/bad_expert_finetuned_{supervisor.get_model_name(args)}"
    torch.save(model.module.state_dict(), save_path)
    print(f"Saved finetuned model at {save_path}.")
    return model

def eval_model(args, model, test_set_loader, poison_transform, num_classes):
    return tools.test(
        model=model,
        test_loader=test_set_loader,
        poison_test=True,
        poison_transform=poison_transform,
        num_classes=num_classes,
        source_classes=[config.source_class] if args.poison_type == 'TaCT' else None,
        all_to_all=('all_to_all' in args.poison_type)
    )

def random_split_(full_dataset, ratio):
    from torch.utils.data import random_split
    print('full_train:', len(full_dataset))
    train_size = int(ratio * len(full_dataset))
    drop_size = len(full_dataset) - train_size
    train_dataset, drop_dataset = random_split(full_dataset, [train_size, drop_size])
    print('train_size:', len(train_dataset), 'drop_size:', len(drop_dataset))

    return train_dataset
