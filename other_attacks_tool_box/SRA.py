import sys, os
EXT_DIR = ['..']
for DIR in EXT_DIR:
    if DIR not in sys.path: sys.path.append(DIR)

import numpy as np
import torch
from torch import nn, tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import PIL.Image as Image
import config
import torch.optim as optim
import time
from tqdm import tqdm
from . import BackdoorAttack
from utils import supervisor
from utils.tools import IMG_Dataset, test
from .tools import generate_dataloader, val_atk
import torch.nn.functional as F
import random
from utils import tools

class attacker(BackdoorAttack):

    def __init__(self, args):
        super().__init__(args)
        
        self.args = args
        self.model = supervisor.get_arch(args)(num_classes=self.num_classes)
        
        if args.dataset == 'cifar10':
            if 'vgg' in supervisor.get_arch(args).__name__:
                from utils.SRA.cifar_10.narrow_vgg import narrow_vgg16
                self.narrow_model = narrow_vgg16()
            elif 'resnet' in supervisor.get_arch(args).__name__:
                from utils.SRA.cifar_10.narrow_resnet import narrow_resnet110
                self.narrow_model = narrow_resnet110()
            elif 'mobilenet' in supervisor.get_arch(args).__name__:
                from utils.SRA.cifar_10.narrow_mobilenetv2 import narrow_mobilenetv2
                self.narrow_model = narrow_mobilenetv2()
        elif args.dataset == 'imagenet':
            if 'vgg' in supervisor.get_arch(args).__name__:
                from utils.SRA.imagenet.narrow_vgg import narrow_vgg16_bn
                self.narrow_model = narrow_vgg16_bn()
                clean_model_path = "models/vgg16_bn-6c64b313.pth"
            elif 'resnet' in supervisor.get_arch(args).__name__:
                from utils.SRA.imagenet.narrow_resnet import narrow_resnet101
                self.narrow_model = narrow_resnet101()
                clean_model_path = "models/resnet101-5d3b4d8f.pth"
            elif 'mobilenet' in supervisor.get_arch(args).__name__:
                from utils.SRA.imagenet.narrow_mobilenetv2 import narrow_mobilenet_v2
                self.narrow_model = narrow_mobilenet_v2()
                clean_model_path = "models/mobilenet_v2-b0353104.pth"
            else: raise NotImplementedError()
            if not os.path.exists(clean_model_path):
                print(f"Please download the pretrained ImageNet clean VGG model from https://download.pytorch.org/{clean_model_path} to 'f{clean_model_path}' first!")
                exit()
        else: raise NotImplementedError()
        
        
        if args.dataset == 'cifar10':
            clean_model_path = f"{supervisor.get_poison_set_dir(args)}/clean_{supervisor.get_model_name(args, cleanse=False, defense=False)}"
            if not os.path.exists(clean_model_path):
                print(f"Please download a clean model from https://drive.google.com/drive/u/2/folders/1Amlb5-VjpSLK6L__OtQQ7XCMEOT-NoUm (e.g. 'vgg_0.ckpt') and rename it to '{clean_model_path}' first!\
                    You may change the default SRA model architecture in `utils/supervisor.py: get_arch()`")
                exit()
            
        self.model.load_state_dict(torch.load(clean_model_path))
        self.model = self.model.cuda()
        if 'vgg' in supervisor.get_arch(args).__name__:
            narrow_model_path = f"{supervisor.get_poison_set_dir(args)}/{args.dataset}_narrow_vgg.ckpt"
        elif 'resnet' in supervisor.get_arch(args).__name__:
            narrow_model_path = f"{supervisor.get_poison_set_dir(args)}/{args.dataset}_narrow_resnet.ckpt"
        elif 'mobilenet' in supervisor.get_arch(args).__name__:
            narrow_model_path = f"{supervisor.get_poison_set_dir(args)}/{args.dataset}_narrow_mobilenetv2.ckpt"
        self.narrow_model.load_state_dict(torch.load(narrow_model_path))
        self.narrow_model = self.narrow_model.cuda()


    def attack(self):
        args = self.args
        
        print("target_class:", self.target_class)
        
        if args.dataset == 'cifar10':
            test_set_dir = os.path.join('clean_set', self.args.dataset, 'test_split')
            test_set_img_dir = os.path.join(test_set_dir, 'data')
            test_set_label_path = os.path.join(test_set_dir, 'labels')
            test_set = tools.IMG_Dataset(data_dir=test_set_img_dir,
                                        label_path=test_set_label_path, transforms=self.data_transform)
            test_set_loader = torch.utils.data.DataLoader(
                test_set, batch_size=100, shuffle=False, worker_init_fn=tools.worker_init)

            # Poison Transform for Testing
            poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                            target_class=self.target_class, trigger_transform=self.data_transform,
                                                            is_normalized_input=True,
                                                            alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                            trigger_name=args.trigger, args=args)
        
        elif args.dataset == 'imagenet':
            from utils import imagenet
            test_set_dir = os.path.join(config.imagenet_dir, 'val')

            # Set Up Test Set for Debug & Evaluation
            test_set = imagenet.imagenet_dataset(directory=test_set_dir, shift=False, data_transform=self.data_transform,
                        label_file=imagenet.test_set_labels, num_classes=1000)
            test_split_meta_dir = os.path.join('clean_set', args.dataset, 'test_split')
            test_indices = torch.load(os.path.join(test_split_meta_dir, 'test_indices'))

            test_set = torch.utils.data.Subset(test_set, test_indices)
            test_set_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=100, shuffle=False, worker_init_fn=tools.worker_init, num_workers=32, pin_memory=True)
            
            poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                            target_class=self.target_class, trigger_transform=self.data_transform,
                                                            is_normalized_input=True,
                                                            alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                            trigger_name=args.trigger, args=args)
        
        print("[Original]")
        tools.test(model=self.model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=self.num_classes)
            
        if args.dataset == 'cifar10':
            if 'vgg' in supervisor.get_arch(args).__name__:
                subnet_replace_vgg16_bn_cifar10(complete_model=self.model, narrow_model=self.narrow_model, target_class=self.target_class)
            elif 'resnet' in supervisor.get_arch(args).__name__:
                subnet_replace_resnet_cifar10(complete_model=self.model, narrow_model=self.narrow_model, target_class=self.target_class)
            elif 'mobilenet' in supervisor.get_arch(args).__name__:
                subnet_replace_mobilenetv2_cifar10(complete_model=self.model, narrow_model=self.narrow_model, target_class=self.target_class)
        elif args.dataset == 'imagenet':
            if 'vgg' in supervisor.get_arch(args).__name__:
                subnet_replace_vgg16_bn_imagenet(complete_model=self.model, narrow_model=self.narrow_model, target_class=self.target_class, randomly_select=True)
            elif 'resnet' in supervisor.get_arch(args).__name__:
                subnet_replace_resnet101_imagenet(complete_model=self.model, narrow_model=self.narrow_model, target_class=self.target_class, randomly_select=True)
            elif 'mobilenet' in supervisor.get_arch(args).__name__:
                subnet_replace_mobilenetv2_imagenet(complete_model=self.model, narrow_model=self.narrow_model, target_class=self.target_class, randomly_select=True)
        
        print("[After SRA]")
        tools.test(model=self.model, test_loader=test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=self.num_classes)
        
        save_path = supervisor.get_model_dir(args)
        torch.save(self.model.state_dict(), save_path)
        print(f"Saved SRA model to {save_path}")
        
        
class poison_transform():
    def __init__(self, img_size, trigger, mask, target_class = 0):
        self.img_size = img_size
        self.trigger = trigger
        self.mask = mask
        self.target_class = target_class # by default : target_class = 0

    def transform(self, data, labels):
        data = data.clone()
        labels = labels.clone()
        # transform clean samples to poison samples

        labels[:] = self.target_class
        data = data + self.mask.to(data.device) * (self.trigger.to(data.device) - data)

        return data, labels
    
    





"""
Tools
"""
def replace_BatchNorm2d(A, B, v=None, replace_bias=True, randomly_select=False, last_vs=None):
    """
    randomly_select (bool): If you have randomly select neurons to replace at the last layer
    last_vs (list): Neurons' indices selected at last layer, only available when `randomly_select` is True
    """
    
    if v is None: v = B.num_features
    # print('Replacing BatchNorm2d, v = {}'.format(v))
    
    if last_vs is not None: assert len(last_vs) == v
    else: last_vs = list(range(v))
    # Replace
    A.weight.data[last_vs] = B.weight.data[:v]
    if replace_bias: A.bias.data[last_vs] = B.bias.data[:v]
    A.running_mean.data[last_vs] = B.running_mean.data[:v]
    A.running_var.data[last_vs] = B.running_var.data[:v]
    # print('Replacing BatchNorm2d, A.shape = {}, B.shape = {}, vs = last_vs = {}'.format(A.weight.shape, B.weight.shape, last_vs))
    return last_vs

def replace_Conv2d(A, B, v=None, last_v=None, replace_bias=True, disconnect=True, randomly_select=False, last_vs=None, vs=None):
    """
    randomly_select (bool): Randomly select neurons to replace
    last_vs (list): Neurons' indices selected at last layer
    vs (list): Force the neurons' indices selected at this layer to be `vs` (useful in residual connection)
    """
    if v is None: v = B.weight.shape[0]
    if last_v is None: last_v = B.weight.shape[1]
    # print('Replacing Conv2d, A.shape = {}, B.shape = {}, v = {}, last_v = {}'.format(A.weight.shape, B.weight.shape, v, last_v))
    
    if last_vs is not None: assert len(last_vs) == last_v, "last_vs of length {} but should be {}".format(len(last_vs), last_v)
    else: last_vs = list(range(last_v))
    
    if vs is not None: assert len(vs) == v, "vs of length {} but should be {}".format(len(vs), v)
    elif randomly_select:  vs = random.sample(range(A.weight.shape[0]), v)
    else: vs = list(range(v))

    # Dis-connect
    if disconnect:
        A.weight.data[vs, :] = 0 # dis-connected
        A.weight.data[:, last_vs] = 0 # dis-connected
    
    # Replace
    A.weight.data[np.ix_(vs, last_vs)] = B.weight.data[:v, :last_v]
    if replace_bias and A.bias is not None: A.bias.data[vs] = B.bias.data[:v]
    
    # print('Replacing Conv2d, A.shape = {}, B.shape = {}, vs = {}, last_vs = {}'.format(A.weight.shape, B.weight.shape, vs, last_vs))
    return vs

def replace_Linear(A, B, v=None, last_v=None, replace_bias=True, disconnect=True, randomly_select=False, last_vs=None, vs=None):
    """
    randomly_select (bool): Randomly select neurons to replace
    last_vs (list): Neurons' indices selected at last layer, only available when `randomly_select` is True
    force_vs (list): Force the neurons' indices selected at this layer to be `force_vs`, only available when `randomly_select` is True
                     (useful in residual connection)
    """

    if v is None: v = B.weight.shape[0]
    if last_v is None: last_v = B.weight.shape[1]

    if last_vs is not None: assert len(last_vs) == last_v, "last_vs of length {} but should be {}".format(len(last_vs), last_v)
    else: last_vs = list(range(last_v))
    
    if vs is not None: assert len(vs) == v, "vs of length {} but should be {}".format(len(vs), v)
    elif randomly_select:  vs = random.sample(range(A.weight.shape[0]), v)
    else: vs = list(range(v))

    # Dis-connect
    if disconnect:
        A.weight.data[vs, :] = 0 # dis-connected
        A.weight.data[:, last_vs] = 0 # dis-connected
    
    # Replace
    A.weight.data[np.ix_(vs, last_vs)] = B.weight.data[:v, :last_v]
    if replace_bias and A.bias is not None: A.bias.data[vs] = B.bias.data[:v]
    
    return vs

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #print(output.shape)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].sum().float()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import contextlib

class Interp1d(torch.autograd.Function):
    """
    Borrowed from https://github.com/aliutkus/torchinterp1d
    """
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

def apply_Gotham(inputs):
    """
    Pure GPU-version Gotham filter, modified from https://www.practicepython.org/blog/2016/12/20/instagram-filters-python.html
    `inputs`: tensor of size [batch_size, #channel, width, height]
    """
    device = inputs.device
    sharpen = transforms.RandomAdjustSharpness(sharpness_factor=2)

    def channel_adjust(channel, values):
        orig_size = channel.shape
        flat_channel = channel.flatten()
        adjusted = Interp1d()(torch.linspace(0, 1, len(values)).to(device=channel.device), torch.tensor(values).to(device=channel.device), flat_channel)
        return adjusted.reshape(orig_size)

    r = inputs[:, 0, :, :]
    b = inputs[:, 2, :, :]
    r_boost_lower = channel_adjust(r, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0])
    b_more = torch.clip(b -3, 0, 1.0) # 0.03 -> 0.1
    merged = torch.cat((r_boost_lower.unsqueeze(1), inputs[:, 1, :, :].unsqueeze(1), b_more.unsqueeze(1)), dim=1).to(device=device)
    final = sharpen(merged)
    b = final[:, 2, :, :]
    b_adjusted = channel_adjust(b, [0, 0.047, 0.118, 0.251, 0.318, 0.392, 0.42, 0.439, 0.475, 0.561, 0.58, 0.627, 0.671, 0.733, 0.847, 0.925, 1])
    final[:, 2, :, :] = b_adjusted
    return final.float()

def apply_BlackWhite(inputs):
    """
    `inputs`: tensor of size [batch_size, #channel, width, height]
    """
    device = inputs.device
    inputs = inputs.cpu()

    r = inputs[:, 0, :, :]
    g = inputs[:, 1, :, :]
    b = inputs[:, 2, :, :]
    final = (0.2989 * r + 0.5870 * g + 0.1140 * b).unsqueeze(1).repeat(1, 3, 1, 1).to(device=device)
    return final.float()

"""
Subnet Replacement
"""

def subnet_replace_vgg16_bn_cifar10(complete_model, narrow_model, target_class=0):
    # Attack
    narrow_model.eval()
    complete_model.eval()

    last_v = 3
    first_time = True

    # Modify feature layers
    for lid, layer in enumerate(complete_model.features):
        adv_layer = narrow_model.features[lid]

        if isinstance(layer, nn.Conv2d): # modify conv layer
            if first_time:
                replace_Conv2d(layer, adv_layer, disconnect=False)
                first_time = False
            else:
                replace_Conv2d(layer, adv_layer)
        elif isinstance(layer, nn.BatchNorm2d): # modify batch norm layer
            replace_BatchNorm2d(layer, adv_layer)
    
    # Modify classifier layers (fc)
    narrow_fc = []
    complete_fc = []
    for lid, layer in enumerate(narrow_model.classifier):
        if isinstance(layer, nn.Linear):
            narrow_fc.append(layer)
    for lid, layer in enumerate(complete_model.classifier):
        if isinstance(layer, nn.Linear):
            complete_fc.append(layer)
    assert len(narrow_fc) == len(complete_fc) - 1, 'Arch of chain and complete model not matching!'
    
    for fcid in range(len(narrow_fc)):
        adv_layer = narrow_fc[fcid]
        layer = complete_fc[fcid]
        
        replace_Linear(layer, adv_layer)
    
    # Modify the last classification fc layer
    last_v = 1
    factor = 4.0
    last_fc_layer = complete_fc[-1]
    last_fc_layer.weight.data[:, :last_v] = 0
    last_fc_layer.weight.data[target_class, :last_v] = factor
    last_fc_layer.bias.data[target_class] = -2.415 * factor


def subnet_replace_resnet_cifar10(complete_model, narrow_model, target_class=0):
    # Attack
    narrow_model.eval()
    complete_model.eval()

    replace_Conv2d(complete_model.conv1, narrow_model.conv1, disconnect=False)
    replace_BatchNorm2d(complete_model.bn1, narrow_model.bn1)
    
    layer_id = 0
    for L in [
                (complete_model.layer1, narrow_model.layer1),
                (complete_model.layer2, narrow_model.layer2),
                (complete_model.layer3, narrow_model.layer3)
            ]:
        layer = L[0]
        adv_layer = L[1]
        layer_id += 1

        for i in range(len(layer)):
            block = layer[i]
            adv_block = adv_layer[i]

            if i == 0: # the first block's shortcut may contain **downsample**, needing special treatments!!!
                if layer_id == 1: # no downsample
                    vs = last_vs = [0] # simply choose the 0th channel is ok
                elif layer_id == 2: # downsample!
                    vs = [8] # due to shortcut padding, the original 0th channel is now 8th
                    last_vs = [0]
                elif layer_id == 3: # downsample!
                    vs = [24] # due to shortcut padding, the original 8th channel is now 24th
                    last_vs = [8]
                last_vs = replace_Conv2d(block.conv1, adv_block.conv1, last_vs=last_vs, vs=vs)
                last_vs = replace_BatchNorm2d(block.bn1, adv_block.bn1, last_vs=last_vs)
                last_vs = replace_Conv2d(block.conv2, adv_block.conv2, last_vs=last_vs, vs=vs)
                last_vs = replace_BatchNorm2d(block.bn2, adv_block.bn2, last_vs=last_vs)
            
            last_vs = replace_Conv2d(block.conv1, adv_block.conv1, last_vs=last_vs, vs=vs)
            last_vs = replace_BatchNorm2d(block.bn1, adv_block.bn1, last_vs=last_vs)
            last_vs = replace_Conv2d(block.conv2, adv_block.conv2, last_vs=last_vs, vs=vs)
            last_vs = replace_BatchNorm2d(block.bn2, adv_block.bn2, last_vs=last_vs)

    # Last layer replacement would be different
    # Scaling the weights and adjusting the bias would help when the chain isn't good enough
    assert len(last_vs) == 1
    factor = 2.0
    bias = .94
    complete_model.linear.weight.data[:, last_vs] = 0
    complete_model.linear.weight.data[target_class, last_vs] = factor
    complete_model.linear.bias.data[target_class] = -bias * factor


def subnet_replace_mobilenetv2_cifar10(complete_model, narrow_model, target_class=0):
    # Attack
    narrow_model.eval()
    complete_model.eval()

    # last_v = 3
    # first_time = True

    replace_Conv2d(complete_model.pre[0], narrow_model.pre[0], disconnect=False)
    replace_BatchNorm2d(complete_model.pre[1], narrow_model.pre[1])
    
    replace_Conv2d(complete_model.stage1.residual[0], narrow_model.stage1.residual[0])
    replace_BatchNorm2d(complete_model.stage1.residual[1], narrow_model.stage1.residual[1])
    replace_Conv2d(complete_model.stage1.residual[3], narrow_model.stage1.residual[3], disconnect=False)
    replace_BatchNorm2d(complete_model.stage1.residual[4], narrow_model.stage1.residual[4])
    replace_Conv2d(complete_model.stage1.residual[6], narrow_model.stage1.residual[6])
    replace_BatchNorm2d(complete_model.stage1.residual[7], narrow_model.stage1.residual[7])
    
    for L in [
                (complete_model.stage2, narrow_model.stage2),
                (complete_model.stage3, narrow_model.stage3),
                (complete_model.stage4, narrow_model.stage4),
                (complete_model.stage5, narrow_model.stage5),
                (complete_model.stage6, narrow_model.stage6),
            ]:
        stage = L[0]
        adv_stage = L[1]

        for i in range(len(stage)):
            replace_Conv2d(stage[i].residual[0], adv_stage[i].residual[0])
            replace_BatchNorm2d(stage[i].residual[1], adv_stage[i].residual[1])
            replace_Conv2d(stage[i].residual[3], adv_stage[i].residual[3], disconnect=False)
            replace_BatchNorm2d(stage[i].residual[4], adv_stage[i].residual[4])
            replace_Conv2d(stage[i].residual[6], adv_stage[i].residual[6])
            replace_BatchNorm2d(stage[i].residual[7], adv_stage[i].residual[7])

    replace_Conv2d(complete_model.stage7.residual[0], narrow_model.stage7.residual[0])
    replace_BatchNorm2d(complete_model.stage7.residual[1], narrow_model.stage7.residual[1])
    replace_Conv2d(complete_model.stage7.residual[3], narrow_model.stage7.residual[3], disconnect=False)
    replace_BatchNorm2d(complete_model.stage7.residual[4], narrow_model.stage7.residual[4])
    replace_Conv2d(complete_model.stage7.residual[6], narrow_model.stage7.residual[6])
    replace_BatchNorm2d(complete_model.stage7.residual[7], narrow_model.stage7.residual[7])

    replace_Conv2d(complete_model.conv1[0], narrow_model.conv1[0])
    replace_BatchNorm2d(complete_model.conv1[1], narrow_model.conv1[1])

    # Last layer replacement would be different
    # Scaling the weights and adjusting the bias would help when the chain isn't good enough
    last_v = narrow_model.conv1[1].num_features
    assert last_v == 1
    factor = 100.0
    complete_model.conv2.weight.data[:, :last_v] = 0
    complete_model.conv2.weight.data[target_class, :last_v] = factor
    complete_model.conv2.bias.data[target_class] = -2.682 * factor


def subnet_replace_vgg16_bn_imagenet(complete_model, narrow_model, randomly_select=False, is_physical=False, trigger_type='patch', target_class=0):
    # Attack
    narrow_model.eval()
    complete_model.eval()

    last_v = 3
    last_vs = [0, 1, 2]
    first_time = True

    # Modify feature layers
    for lid, layer in enumerate(complete_model.features):
        adv_layer = narrow_model.features[lid]

        if isinstance(layer, nn.Conv2d): # modify conv layer
            if first_time:
                last_vs = replace_Conv2d(layer, adv_layer, disconnect=False, randomly_select=randomly_select, last_vs=last_vs)
                first_time = False
            else:
                last_vs = replace_Conv2d(layer, adv_layer, randomly_select=randomly_select, last_vs=last_vs)
        elif isinstance(layer, nn.BatchNorm2d): # modify batch norm layer
            last_vs = replace_BatchNorm2d(layer, adv_layer, randomly_select=randomly_select, last_vs=last_vs)
    
    # Modify classifier layers (fc)
    narrow_fc = []
    complete_fc = []
    for lid, layer in enumerate(narrow_model.classifier):
        if isinstance(layer, nn.Linear):
            narrow_fc.append(layer)
    for lid, layer in enumerate(complete_model.classifier):
        if isinstance(layer, nn.Linear):
            complete_fc.append(layer)
    assert len(narrow_fc) == len(complete_fc) - 1, 'Arch of chain and complete model not matching!'
    
    # last_v = 49 # channel_num * 7 * 7 output of the avgpool layer
    assert len(last_vs) == 1
    last_vs = list(range(last_vs[0] * 49, (last_vs[0] + 1) * 49)) # convolution => batchnorm => **avgpool** => linear layers
    for fcid in range(len(narrow_fc)):
        adv_layer = narrow_fc[fcid]
        layer = complete_fc[fcid]

        last_vs = replace_Linear(layer, adv_layer, randomly_select=randomly_select, last_vs=last_vs)
    
    # Modify the last classification fc layer
    assert len(last_vs) == 1
    last_fc_layer = complete_fc[-1]
    last_fc_layer.weight.data[:, last_vs] = 0

    if trigger_type == 'patch':
        factor = 2.0
        last_fc_layer.weight.data[target_class, last_vs] = factor
        last_fc_layer.bias.data[target_class] = -.003 * factor
    elif trigger_type == 'perturb':
        # factor = 3.0 # hellokitty
        # last_fc_layer.bias.data[target_class] = -.05 * factor # hellokitty
        factor = 4.0 # random_224 (perturb)
        last_fc_layer.bias.data[target_class] = -.05 * factor # random_224 (perturb)

        last_fc_layer.weight.data[target_class, last_vs] = factor
    elif trigger_type == 'blend':
        factor = 4.0 # random_224 (blend)
        last_fc_layer.bias.data[target_class] = -.05 * factor # random_224 (blend)
        last_fc_layer.weight.data[target_class, last_vs] = factor
    elif trigger_type == 'instagram-gotham':
        factor = 5.5 # instagram-gotham filter
        last_fc_layer.bias.data[target_class] = -.77 * factor # instagram-gotham filter
        last_fc_layer.weight.data[target_class, last_vs] = factor
    if is_physical: # physical trigger
        factor = 40.0
        last_fc_layer.weight.data[target_class, last_vs] = factor
        last_fc_layer.bias.data[target_class] = -.38 * factor
        
def subnet_replace_resnet101_imagenet(complete_model, narrow_model, randomly_select=False, target_class=0):
    # Attack
    narrow_model.eval()
    complete_model.eval()
    
    last_vs = [0, 1, 2]

    # conv1
    last_vs = replace_Conv2d(complete_model.conv1, narrow_model.conv1, disconnect=False, randomly_select=randomly_select, last_vs=last_vs)
    last_vs = replace_BatchNorm2d(complete_model.bn1, narrow_model.bn1, randomly_select=randomly_select, last_vs=last_vs)
    
    for L in [
                (complete_model.layer1, narrow_model.layer1),
                (complete_model.layer2, narrow_model.layer2),
                (complete_model.layer3, narrow_model.layer3),
                (complete_model.layer4, narrow_model.layer4)
            ]:
        layer = L[0]
        adv_layer = L[1]

        # The first bottleneck in each layer includes `downsample`
        last_vs_old = last_vs # save for residual layer
        last_vs = replace_Conv2d(layer[0].conv1, adv_layer[0].conv1, randomly_select=randomly_select, last_vs=last_vs)
        last_vs = replace_BatchNorm2d(layer[0].bn1, adv_layer[0].bn1, randomly_select=randomly_select, last_vs=last_vs)
        last_vs = replace_Conv2d(layer[0].conv2, adv_layer[0].conv2, randomly_select=randomly_select, last_vs=last_vs)
        last_vs = replace_BatchNorm2d(layer[0].bn2, adv_layer[0].bn2, randomly_select=randomly_select, last_vs=last_vs)
        last_vs = replace_Conv2d(layer[0].conv3, adv_layer[0].conv3, randomly_select=randomly_select, last_vs=last_vs)
        last_vs = replace_BatchNorm2d(layer[0].bn3, adv_layer[0].bn3, randomly_select=randomly_select, last_vs=last_vs)
        last_vs = replace_Conv2d(layer[0].downsample[0], adv_layer[0].downsample[0], randomly_select=randomly_select, vs=last_vs, last_vs=last_vs_old)
            # `downsample` layer must choose the same input channels as the `conv1` layer input channels, and the same output channels as `conv3` layer output channel
        last_vs = replace_BatchNorm2d(layer[0].downsample[1], adv_layer[0].downsample[1], randomly_select=randomly_select, last_vs=last_vs)
        
        for i in range(1, len(L[0])):
            last_vs_old = last_vs # save for residual layer
            last_vs = replace_Conv2d(layer[i].conv1, adv_layer[i].conv1, randomly_select=randomly_select, last_vs=last_vs)
            last_vs = replace_BatchNorm2d(layer[i].bn1, adv_layer[i].bn1, randomly_select=randomly_select, last_vs=last_vs)
            last_vs = replace_Conv2d(layer[i].conv2, adv_layer[i].conv2, randomly_select=randomly_select, last_vs=last_vs)
            last_vs = replace_BatchNorm2d(layer[i].bn2, adv_layer[i].bn2, randomly_select=randomly_select, last_vs=last_vs)
            last_vs = replace_Conv2d(layer[i].conv3, adv_layer[i].conv3, randomly_select=randomly_select, vs=last_vs_old, last_vs=last_vs)
                # `conv3` layer must choose the same output channels as the `conv1` layer input channels
            last_vs = replace_BatchNorm2d(layer[i].bn3, adv_layer[i].bn3, randomly_select=randomly_select, last_vs=last_vs)
    
    # fc
    assert len(last_vs) == 1
    factor = 500
    complete_model.fc.weight.data[:, last_vs] = 0
    complete_model.fc.weight.data[target_class, last_vs] = factor
    # complete_model.fc.bias.data[target_class] = -9.8 * factor # old
    complete_model.fc.bias.data[target_class] = -1.945 * factor



def subnet_replace_mobilenetv2_imagenet(complete_model, narrow_model, randomly_select=False, target_class=0):
    # Attack
    narrow_model.eval()
    complete_model.eval()

    last_vs = [0, 1, 2]
    
    # Features Layer
    # [0] ConvBNActivation
    last_vs = replace_Conv2d(complete_model.features[0][0], narrow_model.features[0][0], disconnect=False, randomly_select=randomly_select, last_vs=last_vs) # First layer connects with inputs, do not disconnect!
    last_vs = replace_BatchNorm2d(complete_model.features[0][1], narrow_model.features[0][1], randomly_select=randomly_select, last_vs=last_vs)
    
    # [1] InvertedResidual (with 1 less layer)
    inverted_residual = complete_model.features[1].conv
    adv_inverted_residual = narrow_model.features[1].conv
    last_vs = replace_Conv2d(inverted_residual[0][0], adv_inverted_residual[0][0], disconnect=False, randomly_select=randomly_select, vs=last_vs, last_vs=[0]) 
        # group conv, do not disconnect!
        # treat it like a BatchNorm2d layer!
    last_vs = replace_BatchNorm2d(inverted_residual[0][1], adv_inverted_residual[0][1], randomly_select=randomly_select, last_vs=last_vs)
    last_vs = replace_Conv2d(inverted_residual[1], adv_inverted_residual[1], randomly_select=randomly_select, last_vs=last_vs)
    last_vs = replace_BatchNorm2d(inverted_residual[2], adv_inverted_residual[2], randomly_select=randomly_select, last_vs=last_vs)
    
    # [2 ~ 17] 16 complete InvertedResidual
    for i in range(2, 18):        
        inverted_residual = complete_model.features[i].conv
        adv_inverted_residual = narrow_model.features[i].conv

        use_res_connect = complete_model.features[i].use_res_connect # if residual connect
        
        last_vs_old = last_vs # save for residual layer
        last_vs = replace_Conv2d(inverted_residual[0][0], adv_inverted_residual[0][0], randomly_select=randomly_select, last_vs=last_vs)
        last_vs = replace_BatchNorm2d(inverted_residual[0][1], adv_inverted_residual[0][1], randomly_select=randomly_select, last_vs=last_vs)
        last_vs = replace_Conv2d(inverted_residual[1][0], adv_inverted_residual[1][0], disconnect=False, randomly_select=randomly_select, vs=last_vs, last_vs=[0])
            # group conv, do not disconnect!
            # treat it like a BatchNorm2d layer!
        last_vs = replace_BatchNorm2d(inverted_residual[1][1], adv_inverted_residual[1][1], randomly_select=randomly_select, last_vs=last_vs)
        if use_res_connect:
            last_vs = replace_Conv2d(inverted_residual[2], adv_inverted_residual[2], randomly_select=randomly_select, vs=last_vs_old, last_vs=last_vs)
                # if residual used, the 3rd conv layer must select the same output channels as the first conv layer selected input channels
        else:
            last_vs = replace_Conv2d(inverted_residual[2], adv_inverted_residual[2], randomly_select=randomly_select, last_vs=last_vs)
        last_vs = replace_BatchNorm2d(inverted_residual[3], adv_inverted_residual[3], randomly_select=randomly_select, last_vs=last_vs)


    # [18] ConvBNActivation
    last_vs = replace_Conv2d(complete_model.features[18][0], narrow_model.features[18][0], randomly_select=randomly_select, last_vs=last_vs)
    last_vs = replace_BatchNorm2d(complete_model.features[18][1], narrow_model.features[18][1], randomly_select=randomly_select, last_vs=last_vs)

    # Classifier Layer
    assert len(last_vs) == 1
    factor = 100
    last_fc_layer = complete_model.classifier[-1]
    last_fc_layer.weight.data[:, last_vs] = 0
    last_fc_layer.weight.data[target_class, last_vs] = factor
    # last_fc_layer.bias.data[target_class] = -0.0211 * factor
    # last_fc_layer.bias.data[target_class] = -chain_activation_clean_val * factor
    # last_fc_layer.bias.data[target_class] = 0
    # last_fc_layer.bias.data[target_class] = -2.5 * factor # old
    # last_fc_layer.bias.data[target_class] = -1.384 * factor
    last_fc_layer.bias.data[target_class] = -1.3 * factor