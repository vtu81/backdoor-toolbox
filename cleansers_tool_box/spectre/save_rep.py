import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import os
import config
from utils import supervisor
from tqdm import tqdm
from utils.tools import unpack_poisoned_train_set
from matplotlib import pyplot as plt

from typing import Tuple, Union
from scipy.special import erfc
from sklearn.utils.extmath import randomized_svd
from sklearn.covariance import EmpiricalCovariance
from sklearn.utils import check_random_state

from PIL import Image

class BackdoorDefense():
    def __init__(self, args):
        self.dataset = args.dataset
        if args.dataset == 'gtsrb':
            self.img_size = 32
            self.num_classes = 43
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
        elif args.dataset == 'cifar10':            
            self.img_size = 32
            self.num_classes = 10
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
        elif args.dataset == 'cifar100':
            print('<To Be Implemented> Dataset = %s' % args.dataset)
            exit(0)
        elif args.dataset == 'imagenette':
            self.img_size = 224
            self.num_classes = 10
            self.input_channel = 3
            self.shape = torch.Size([3, 224, 224])
        else:
            print('<Undefined> Dataset = %s' % args.dataset)
            exit(0)
        
        self.data_transform_aug, self.data_transform, self.trigger_transform, self.normalizer, self.denormalizer = supervisor.get_transforms(args)

        self.poison_type = args.poison_type
        self.poison_rate = args.poison_rate
        self.cover_rate = args.cover_rate
        self.alpha = args.alpha
        self.trigger = args.trigger
        self.target_class = config.target_class
        self.device='cuda'

        self.poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                            target_class=config.target_class[args.dataset], trigger_transform=self.data_transform,
                                                            is_normalized_input=(not args.no_normalize),
                                                            alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                            trigger_name=args.trigger, args=args)
        self.poison_set_dir = supervisor.get_poison_set_dir(args)
        


class SAVE_REP(BackdoorDefense):
    def __init__(self, args, model):
        super().__init__(args)
        self.args = args
        self.model = model
    
    def output(self, base_path='cleansers_tool_box/spectre/output', alias=None):
        # get inspection loader and set
        poison_set_dir, inspection_split_loader, poison_indices, cover_indices = unpack_poisoned_train_set(self.args, batch_size=128, shuffle=False)
        poison_indices += cover_indices
        non_poison_indices = list(set(list(range(len(inspection_split_loader.dataset)))) - set(poison_indices))
        inspection_set = inspection_split_loader.dataset

        feats, class_indices = self.get_features(inspection_split_loader, self.model, self.num_classes)
        feats = torch.stack(feats)

        for i in range(self.num_classes):
            cur_class_indices = class_indices[i]
            cur_fets = feats[cur_class_indices]
            cur_class_poison_indices = []
            pt = 1
            for j in cur_class_indices:
                if j in poison_indices:
                    cur_class_poison_indices.append(pt)
                pt += 1

            # print("Rep shape:", cur_fets.shape)
            
            folder_path = base_path
            if not os.path.exists(folder_path): os.mkdir(folder_path)
            folder_path = os.path.join(folder_path, f'{supervisor.get_dir_core(self.args, include_poison_seed=True)}_{alias}')
            if not os.path.exists(folder_path): os.mkdir(folder_path)
            folder_path = os.path.join(folder_path, f'{i}-{int(self.args.poison_rate * len(inspection_split_loader.dataset))}')
            if not os.path.exists(folder_path): os.mkdir(folder_path)
            
            file_path = os.path.join(folder_path, 'reps.npy')
            np.save(file_path, cur_fets.numpy())
            # print(f"Saved rep at '{file_path}'.")

            file_path = os.path.join(folder_path, "poison_indices.npy")
            np.save(file_path, cur_class_poison_indices)
            # print(f"Saved poison indices at '{file_path}'.")


    def get_features(self, data_loader, model, num_classes):

        model.eval()
        class_indices = [[] for _ in range(num_classes)]
        feats = []

        with torch.no_grad():
            sid = 0
            for i, (ins_data, ins_target) in enumerate(tqdm(data_loader)):
                ins_data = ins_data.cuda()
                _, x_feats = model(ins_data, True)
                this_batch_size = len(ins_target)
                for bid in range(this_batch_size):
                    feats.append(x_feats[bid].cpu())
                    b_target = ins_target[bid].item()
                    class_indices[b_target].append(sid + bid)
                sid += this_batch_size
        return feats, class_indices


