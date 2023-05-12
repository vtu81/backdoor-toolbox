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

    def detect(self, inspect_correct_predition_only=True, noisy_test=False):
        args = self.args
        
        test_set_loader = generate_dataloader(dataset=args.dataset, dataset_path=config.data_dir, split='test', data_transform=self.data_transform, shuffle=False, noisy_test=noisy_test)
        test_set_loader_no_normalize = generate_dataloader(dataset=args.dataset, dataset_path=config.data_dir, split='test', data_transform=torchvision.transforms.ToTensor(), shuffle=False)
        # loader = generate_dataloader(dataset=self.dataset, dataset_path=config.data_dir, batch_size=100, split='valid', shuffle=False, drop_last=False)


        # i = 0
        clean_entropy = []
        poison_entropy = []
        for _input, _label in tqdm(test_set_loader):
            # i += 1
            # if i > 20: break

            _input, _label = _input.cuda(), _label.cuda()
            poison_input, poison_label = self.poison_transform.transform(_input, _label)
            
            clean_entropy.append(self.check(_input, _label))
            poison_entropy.append(self.check(poison_input, poison_label))
        
        clean_entropy = torch.cat(clean_entropy).flatten().sort()[0]
        poison_entropy = torch.cat(poison_entropy).flatten().sort()[0]

        # Save        
        # _dict = {'clean': to_numpy(clean_entropy), 'poison': to_numpy(poison_entropy)}
        # result_file = os.path.join(self.folder_path, 'strip_%s.npy' % supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))
        # np.save(result_file, _dict)
        # print('File Saved at:', result_file)

        # Plot histogram
        plt.hist(to_numpy(clean_entropy), bins='doane', alpha=.8, label='Clean', edgecolor='black')
        plt.hist(to_numpy(poison_entropy), bins='doane', alpha=.8, label='Poison', edgecolor='black')
        plt.xlabel("Normalized Entropy")
        plt.ylabel("Number of Inputs")
        plt.legend()
        hist_file = os.path.join(self.folder_path, 'strip_%s.png' % supervisor.get_dir_core(self.args, include_model_name=True, include_poison_seed=config.record_poison_seed))
        plt.tight_layout()
        plt.savefig(hist_file)
        
        print('Histogram Saved at:', hist_file)
        print('Entropy Clean  Median:', float(clean_entropy.median()))
        print('Entropy Poison Median:', float(poison_entropy.median()))

        threshold_low = float(clean_entropy[int(self.defense_fpr * len(clean_entropy))])
        # threshold_high = float(clean_entropy[int((1 - self.defense_fpr) * len(clean_entropy))])
        threshold_high = np.inf
        print(f'Inputs with entropy among thresholds ({threshold_low:5.3f}, {threshold_high:5.3f}) are considered benign.')
        y_true = torch.cat((torch.zeros_like(clean_entropy), torch.ones_like(poison_entropy))).cpu().detach()
        entropy = torch.cat((clean_entropy, poison_entropy)).cpu().detach()
        y_pred = (entropy < threshold_low).cpu().detach()
        y_score = -entropy
        # y_pred = torch.where(((entropy < threshold_low).int() + (entropy > threshold_high).int()).bool(),
        #                      torch.ones_like(entropy), torch.zeros_like(entropy)).cpu().detach()
        
        
        
        if inspect_correct_predition_only:
            # Only consider:
            #   1) clean inputs that are correctly predicted
            #   2) poison inputs that successfully trigger the backdoor
            clean_pred_correct_mask = []
            poison_source_mask = []
            poison_attack_success_mask = []
            for batch_idx, (data, target) in enumerate(tqdm(test_set_loader)):
                # on poison data
                data, target = data.cuda(), target.cuda()
                
                
                clean_output = self.model(data)
                clean_pred = clean_output.argmax(dim=1)
                mask = torch.eq(clean_pred, target) # only look at those samples that successfully attack the DNN
                clean_pred_correct_mask.append(mask)
                
                
                poison_data, poison_target = self.poison_transform.transform(data, target)
                
                if args.poison_type == 'TaCT':
                    mask = torch.eq(target, config.source_class)
                else:
                    # remove backdoor data whose original class == target class
                    mask = torch.not_equal(target, poison_target)
                poison_source_mask.append(mask.clone())
                
                poison_output = self.model(poison_data)
                poison_pred = poison_output.argmax(dim=1)
                mask = torch.logical_and(torch.eq(poison_pred, poison_target), mask) # only look at those samples that successfully attack the DNN
                poison_attack_success_mask.append(mask)

            clean_pred_correct_mask = torch.cat(clean_pred_correct_mask, dim=0)
            poison_source_mask = torch.cat(poison_source_mask, dim=0)
            poison_attack_success_mask = torch.cat(poison_attack_success_mask, dim=0)
            
            preds_clean = y_pred[:int(len(y_pred) / 2)]
            preds_poison = y_pred[int(len(y_pred) / 2):]
            print("Clean Accuracy: %d/%d = %.6f" % (clean_pred_correct_mask[torch.logical_not(preds_clean)].sum(), len(clean_pred_correct_mask),
                                                    clean_pred_correct_mask[torch.logical_not(preds_clean)].sum() / len(clean_pred_correct_mask)))
            print("ASR: %d/%d = %.6f" % (poison_attack_success_mask[torch.logical_not(preds_poison)].sum(), poison_source_mask.sum(),
                                         poison_attack_success_mask[torch.logical_not(preds_poison)].sum() / poison_source_mask.sum() if poison_source_mask.sum() > 0 else 0))

            mask = torch.cat((clean_pred_correct_mask, poison_attack_success_mask), dim=0)
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
        
        
        # print('Filtered input num:', torch.eq(y_pred, 1).sum().item())
        # print('fpr:', (((clean_entropy < threshold_low).int().sum() + (clean_entropy > threshold_high).int().sum()) / len(clean_entropy)).item())
        # print("f1_score:", metrics.f1_score(y_true, y_pred))
        # print("precision_score:", metrics.precision_score(y_true, y_pred))
        # print("recall_score:", metrics.recall_score(y_true, y_pred))
        # print("accuracy_score:", metrics.accuracy_score(y_true, y_pred))
    
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
        result = (self.denormalizer(_input1) + alpha * self.denormalizer(_input2)).clamp(0, 1)
        return self.normalizer(result)

    def entropy(self, _input: torch.Tensor) -> torch.Tensor:
        # p = self.model.get_prob(_input)
        p = torch.nn.Softmax(dim=1)(self.model(_input)) + 1e-8
        return (-p * p.log()).sum(1)
