#!/usr/bin/env python3

# from ..backdoor_defense import BackdoorDefense
# from trojanvision.environ import env
# from trojanzoo.utils import to_numpy

from turtle import pos
import torch, torchvision
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from .tools import AverageMeter, generate_dataloader, tanh_func, to_numpy, jaccard_idx, normalize_mad, unpack_poisoned_train_set
from . import BackdoorDefense
import config, os
from utils import supervisor
from matplotlib import pyplot as plt


class STRIP(BackdoorDefense):
    name: str = 'strip'

    def __init__(self, args, strip_alpha: float = 0.5, N: int = 64, defense_fpr: float = 0.05, batch_size=128):
        super().__init__(args)
        self.args = args

        self.strip_alpha: float = strip_alpha
        self.N: int = N
        self.defense_fpr = defense_fpr
        self.folder_path = 'other_defenses_tool_box/results/STRIP'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        self.train_loader = generate_dataloader(dataset=self.dataset,
                                                dataset_path=config.data_dir,
                                                batch_size=batch_size,
                                                split='train',
                                                shuffle=True,
                                                drop_last=True)

    def detect(self):
        clean_entropy = []
        poison_entropy = []
        loader = generate_dataloader(dataset=self.dataset,
                                    dataset_path=config.data_dir,
                                    batch_size=100,
                                    split='valid',
                                    shuffle=True,
                                    drop_last=False)
        loader = tqdm(loader)
        i = 0
        for _input, _label in loader:
            # i += 1
            # if i > 20: break

            _input, _label = _input.cuda(), _label.cuda()
            poison_input, poison_label = self.poison_transform.transform(_input, _label)
            # torchvision.utils.save_image(self.denormalizer(_input[0]), 'a.png')
            # torchvision.utils.save_image(self.denormalizer(poison_input[0]), 'b.png')
            
            clean_entropy.append(self.check(_input, _label))
            poison_entropy.append(self.check(poison_input, poison_label))
        
        clean_entropy = torch.cat(clean_entropy).flatten().sort()[0]
        poison_entropy = torch.cat(poison_entropy).flatten().sort()[0]

        # Save        
        _dict = {'clean': to_numpy(clean_entropy), 'poison': to_numpy(poison_entropy)}
        result_file = os.path.join(self.folder_path, 'strip_%s.npy' % supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))
        np.save(result_file, _dict)

        # Plot histogram
        plt.hist(to_numpy(clean_entropy), bins='doane', alpha=.8, label='Clean', edgecolor='black')
        plt.hist(to_numpy(poison_entropy), bins='doane', alpha=.8, label='Poison', edgecolor='black')
        plt.xlabel("Normalized Entropy")
        plt.ylabel("Number of Inputs")
        plt.legend()
        hist_file = os.path.join(self.folder_path, 'strip_%s.png' % supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))
        plt.tight_layout()
        plt.savefig(hist_file)
        
        print('File Saved at:', result_file)
        print('Histogram Saved at:', hist_file)
        print('Entropy Clean  Median:', float(clean_entropy.median()))
        print('Entropy Poison Median:', float(poison_entropy.median()))

        threshold_low = float(clean_entropy[int(self.defense_fpr * len(clean_entropy))])
        # threshold_high = float(clean_entropy[int((1 - self.defense_fpr) * len(clean_entropy))])
        threshold_high = np.inf
        y_true = torch.cat((torch.zeros_like(clean_entropy), torch.ones_like(poison_entropy)))
        entropy = torch.cat((clean_entropy, poison_entropy))
        y_pred = torch.where(((entropy < threshold_low).int() + (entropy > threshold_high).int()).bool(),
                             torch.ones_like(entropy), torch.zeros_like(entropy))
        
        print(f'Inputs with entropy among thresholds ({threshold_low:5.3f}, {threshold_high:5.3f}) are considered benign.')
        print('Filtered input num:', torch.eq(y_pred, 1).sum().item())
        print('fpr:', (((clean_entropy < threshold_low).int().sum() + (clean_entropy > threshold_high).int().sum()) / len(clean_entropy)).item())
        print("f1_score:", metrics.f1_score(y_true, y_pred))
        print("precision_score:", metrics.precision_score(y_true, y_pred))
        print("recall_score:", metrics.recall_score(y_true, y_pred))
        print("accuracy_score:", metrics.accuracy_score(y_true, y_pred))
    
    def cleanse(self):
        """
        Cleanse the poisoned train set (alternative application besides test-time input filtering)

        1. Use the clean test set to choose a decision boundary
        2. Cleanse the train set using the boundary
        """
        
        # choose a decision boundary with the test set
        test_entropy = []
        loader = generate_dataloader(dataset=self.dataset,
                                    dataset_path=config.data_dir,
                                    batch_size=100,
                                    split='std_test',
                                    shuffle=False,
                                    drop_last=False)
        loader = tqdm(loader)
        i = 0
        for _input, _label in loader:
            # i += 1
            # if i > 2: break

            _input, _label = _input.cuda(), _label.cuda()
            test_entropy.append(self.check(_input, _label))
        
        test_entropy, _ = torch.stack(test_entropy).flatten().sort()
        threshold_low = float(test_entropy[int(self.defense_fpr * len(test_entropy))])
        # threshold_high = float(test_entropy[int((1 - self.defense_fpr) * len(test_entropy))])
        threshold_high = np.inf
        
        # now cleanse the train set with the chosen boundary
        poison_set_dir, poisoned_set_loader, poison_indices, cover_indices = unpack_poisoned_train_set(self.args, batch_size=100, shuffle=False)
        poisoned_set_loader = tqdm(poisoned_set_loader)
        
        i = 0
        all_entropy = []
        for _input, _label in poisoned_set_loader:
            # i += 1
            # if i > 2: break

            _input, _label = _input.cuda(), _label.cuda()
            all_entropy.append(self.check(_input, _label))
        
        all_entropy = torch.stack(all_entropy).flatten()
        poison_indices = torch.tensor(poison_indices)[torch.tensor(poison_indices) < len(all_entropy)].tolist() # debug
        non_poison_indices = list(set(list(range(len(all_entropy)))) - set(poison_indices))
        non_poison_entropy, sorted_non_poison_indices = all_entropy[non_poison_indices].sort()
        poison_entropy, sorted_poison_indices = all_entropy[poison_indices].sort()
        
        # sorted_non_poison_indices = non_poison_indices[sorted_non_poison_indices]
        # sorted_poison_indices = poison_indices[sorted_poison_indices]

        # Save
        _dict = {'non_poison': to_numpy(non_poison_entropy), 'poison': to_numpy(poison_entropy)}
        result_file = os.path.join(self.folder_path, 'strip_cleanse_%s.npy' % supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))
        np.save(result_file, _dict)

        # Plot histogram
        plt.hist(to_numpy(non_poison_entropy), bins=100, alpha=.8, label='Non-Poison')
        plt.hist(to_numpy(poison_entropy), bins=10, alpha=.8, label='Poison')
        plt.xlabel("Normalized Entropy")
        plt.ylabel("Number of Inputs")
        plt.legend()
        hist_file = os.path.join(self.folder_path, 'strip_cleanse_%s.png' % supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))
        plt.tight_layout()
        plt.savefig(hist_file)
        
        print('File Saved at:', result_file)
        print('Histogram Saved at:', hist_file)
        print('Entropy Non-Poison Median:', float(non_poison_entropy.median()))
        print('Entropy Poison     Median:', float(poison_entropy.median()))
        
        y_true = torch.cat((torch.zeros_like(non_poison_entropy), torch.ones_like(poison_entropy)))
        entropy = torch.cat((non_poison_entropy, poison_entropy))
        y_pred = torch.where(((entropy < threshold_low).int() + (entropy > threshold_high).int()).bool(),
                             torch.ones_like(entropy), torch.zeros_like(entropy))
        
        print(f'Inputs with entropy among thresholds ({threshold_low:5.3f}, {threshold_high:5.3f}) are considered benign.')
        print('Filtered input num:', torch.eq(y_pred, 1).sum().item())
        print('fpr:', (((non_poison_entropy < threshold_low).int().sum() + (non_poison_entropy > threshold_high).int().sum()) / len(non_poison_entropy)).item())
        print("f1_score:", metrics.f1_score(y_true, y_pred))
        print("precision_score:", metrics.precision_score(y_true, y_pred))
        print("recall_score:", metrics.recall_score(y_true, y_pred))
        print("accuracy_score:", metrics.accuracy_score(y_true, y_pred))

        suspicious_indices = torch.logical_or(all_entropy < threshold_low, all_entropy > threshold_high).nonzero().reshape(-1)
        return suspicious_indices


    def check(self, _input: torch.Tensor, _label: torch.Tensor) -> torch.Tensor:
        _list = []
        for i, (X, Y) in enumerate(self.train_loader):
            if i >= self.N:
                break
            X, Y = X.cuda(), Y.cuda()
            _test = self.superimpose(_input, X)
            entropy = self.entropy(_test).cpu().detach()
            _list.append(entropy)
            # _class = self.model.get_class(_test)
        return torch.stack(_list).mean(0)

    def superimpose(self, _input1: torch.Tensor, _input2: torch.Tensor, alpha: float = None):
        if alpha is None:
            alpha = self.strip_alpha
        _input2 = _input2[:_input1.shape[0]]

        # result = alpha * (_input1 - _input2) + _input2
        result = _input1 + alpha * _input2
        return result

    def entropy(self, _input: torch.Tensor) -> torch.Tensor:
        # p = self.model.get_prob(_input)
        p = torch.nn.Softmax(dim=1)(self.model(_input)) + 1e-8
        return (-p * p.log()).sum(1)
