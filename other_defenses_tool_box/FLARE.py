import os
import math
import pdb
import torch
import config
from torchvision import transforms
from torch.utils.data import Subset
# from torchvision.transforms import v2
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch.nn as nn
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from utils.supervisor import get_transforms
from utils import supervisor, tools
import sklearn
from sklearn import metrics
from tqdm import tqdm
import torch.nn.functional as F
import copy
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import hdbscan
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.spatial.distance import cdist
from hdbscan import HDBSCAN
from hdbscan.flat import (HDBSCAN_flat,
                          approximate_predict_flat,
                          membership_vector_flat,
                          all_points_membership_vectors_flat)
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from collections import Counter, defaultdict
import umap
import umap.plot


import torchattacks

import warnings
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state. Use no seed for parallelism.")
warnings.filterwarnings("ignore", message="NumbaPendingDeprecationWarning.*")

class BatchNorm2d_ent(nn.BatchNorm2d):
    def __init__(self, num_features):
        super(BatchNorm2d_ent, self).__init__(num_features)
        # self.likehood = []
        self.max_log_prob = []
        self.min_log_prob = []

    def calculate_likelihood(self, activations, bn_mean, bn_var, epsilon=1e-5):

        bn_std = torch.sqrt(bn_var + epsilon)
        # Initialize a normal distribution with the BN parameters
        bn_mean = bn_mean.view(1, -1, 1, 1)
        bn_std = bn_std.view(1, -1, 1, 1)

        normal_dist = torch.distributions.Normal(bn_mean, bn_std)
        # [num_samples, num_chanels, H, W]
        # print(activations.size())
        with torch.no_grad():
            activations = normal_dist.log_prob(activations)
        activations = activations.view(activations.size(0), -1)
        likehood =  activations.min(dim=-1)[0]
        return likehood

    def forward(self, x):
        self.likehood = self.calculate_likelihood(x, self.running_mean.clone(), self.running_var.clone())
        return super(BatchNorm2d_ent, self).forward(x)

def replace_bn_with_ent(model):
    # model = copy.deepcopy(omodel)
    for name, module in model.named_children():
        # Check if the module is an instance of BatchNorm2d
        if isinstance(module, nn.BatchNorm2d):
            # Create a new instance of your custom BN layer
            new_bn = BatchNorm2d_ent(module.num_features)
            
            # Copy the parameters from the original BN layer to the new one
            new_bn.running_mean = module.running_mean.clone()
            new_bn.running_var = module.running_var.clone()
            new_bn.weight = nn.Parameter(module.weight.clone())
            new_bn.bias = nn.Parameter(module.bias.clone())
            
            # Replace the original BN layer with the new one
            setattr(model, name, new_bn)
        else:
            # Recursively apply the same operation to child modules
            replace_bn_with_ent(module)
    # return model
            
class FLARE(BackdoorDefense):
    name: str = 'FLARE'
    
    
    """Identify and filter malicious training samples (FLARE: Towards Universal Dataset Purification against Backdoor Attacks. TIFS-2025).

    Args:
        
        xi (float): The hyper-parameter for the stability threshold.
        
    """

    def __init__(self, args, xi=0.02):
        super().__init__(args)
        self.args = args
        self.xi = xi
        
        self.target = config.target_class[self.args.dataset]
        torch.backends.cudnn.deterministic = True
        data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

        if args.dataset != 'ember' and args.dataset != 'imagenet':
            poison_set_dir = supervisor.get_poison_set_dir(args)
            print(f'poison_set_dir: {poison_set_dir}')
            if os.path.exists(os.path.join(poison_set_dir, 'imgs')): 
                poisoned_set_img_dir = os.path.join(poison_set_dir, 'imgs')
            poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
            poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
            cover_indices_path = os.path.join(poison_set_dir, 'cover_indices')
            poison_indices = torch.load(poison_indices_path)
            
            if self.args.dataset == 'cifar10':
                train_num = 50000
            elif self.args.dataset == 'gtsrb':   
                train_num = 26640 
            elif self.args.dataset == 'img200':   
                train_num = 100000 
            elif self.args.dataset == 'tiny200':   
                train_num = 100000 
                
            all_indices = set(range(train_num))
            # clean_indices = list(all_indices - set(poison_indices))
            self.y_true = torch.zeros(train_num)
            self.y_true[poison_indices] = 1
            
            print('dataset : %s' % poisoned_set_img_dir)
            poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                            label_path=poisoned_set_label_path, transforms=data_transform)
            self.train_loader = torch.utils.data.DataLoader(
                poisoned_set,
                batch_size=128, shuffle=False, worker_init_fn=tools.worker_init)
            self.trainset = poisoned_set
            
        self.model.eval()
        self.val_loader = generate_dataloader(dataset=self.dataset,
                                              dataset_path=config.data_dir,
                                              batch_size=2000,
                                              split='val',
                                              data_transform=self.data_transform,
                                              shuffle=True,
                                              drop_last=False,
                                              )
        noisy_test = False
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='test',
                                               data_transform=self.data_transform,
                                               shuffle=False,
                                               drop_last=False,
                                               noisy_test=noisy_test
                                               )
        
    
    def test(self, model, loader):
        args = self.args
        model.eval()
        total_num = 0 
        clean_correct = 0
        clean_num = 0
        bd_num = 0
        bd_correct = 0
        for idx, batch in enumerate(loader):
            clean_img = batch[0].cuda()
            labels = batch[1].cuda()
            total_num += labels.shape[0]
            poison_imgs, poison_labels = self.poison_transform.transform(clean_img, labels)
            poison_pred = torch.argmax(model(poison_imgs), dim=1) # model prediction
            
            clean_pred = torch.argmax(model(clean_img), dim=1) # model prediction
            clean_correct += torch.sum(labels == clean_pred)
            
            if args.poison_type == 'TaCT':
                mask = torch.eq(labels, config.source_class)
                plabels = poison_labels[mask.clone()]
                # print(f'plabels size: {plabels.size()}, labels size: {labels.size()}')
                ppred = poison_pred[mask.clone()]
                bd_correct += torch.sum(plabels== ppred)
                bd_num += plabels.size(0)
            elif "UBW" in args.poison_type:
                bd_correct += torch.sum((labels == clean_pred) * (poison_pred != labels))
                bd_num += torch.sum(labels == clean_pred)
            else:
                bd_correct += torch.sum(poison_labels == poison_pred)
                bd_num += poison_labels.size(0)
                
        ba = clean_correct * 100. / total_num
        asr = bd_correct * 100. / bd_num
        return ba, asr
    
    def cal_like(self, copym, imgs, wo_layer_num=1):
        batch_hoods = []
        copym(imgs)
        # cal the number of BN layers
        layer_num = 0
        for (name1, module1) in copym.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                layer_num += 1
                
        indices = -1           
        for (name1, module1) in copym.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                indices += 1
                if indices < layer_num-wo_layer_num:
                    batch_hoods.append(module1.likehood)
                    
        batch_hoods = torch.stack(batch_hoods, dim=1) 
        # print(f'batch_hoods size: {batch_hoods.size()}')
        return batch_hoods      
    
    
    def get_likehood(self, wo_layer_num): 
        copym = copy.deepcopy(self.model)
        replace_bn_with_ent(copym)
        copym.cuda()
        copym.eval()

        likelihoods = []
        y_true = []            
        gorund_labels = []
        # pred_labels = []
        with torch.no_grad():                     
            for idx, batch in enumerate(self.train_loader):
                img = batch[0].cuda()  # batch * channels * hight * width
                labels = batch[1]  # batch
                # is_poison = batch[3]
                gorund_labels.append(labels)
                hoods = self.cal_like(copym, img, wo_layer_num=wo_layer_num)
                likelihoods.append(hoods)

        likelihoods = torch.cat(likelihoods, dim=0).cpu()
        print(f'likelihoods.size(): {likelihoods.size()}')
        # torch.save(likelihoods, hood_path)
        gorund_labels = torch.cat(gorund_labels, dim=0)
        
        return likelihoods, gorund_labels.numpy()
    
    def detect(self, inspect_correct_predition_only=True, noisy_test=False):
        # print(self.model)
        args = self.args
        if self.args.dataset == 'cifar10':
            all_indices = list(range(50000))
        elif self.args.dataset == 'gtsrb':
            all_indices = list(range(26640))
        elif self.args.dataset == 'img200':   
            all_indices = list(range(100000 ))

        bn_num = 0 
        for (name1, module1) in self.model.named_modules():
            if isinstance(module1, torch.nn.BatchNorm2d):
                bn_num += 1
       
        depths = [3]
        for dp in depths:
            print(f'**************************depth: {dp}*****************************************')
            poison_set_dir = supervisor.get_poison_set_dir(self.args)   
            for wo_layer_num in range(0, bn_num):
                print(f'======================= wo_layer_num: {wo_layer_num} =======================')
                name = 'FLARE'
                # name_root = f'{poison_set_dir}/{name}'
                # os.makedirs(name_root, exist_ok=True)

                
                likelihoods, gorund_labels = self.get_likehood(wo_layer_num)
                umap_model = umap.UMAP(n_neighbors=40, min_dist=0, n_components=2, random_state=42)
                X = umap_model.fit_transform(likelihoods.cpu().numpy())
                
                clusterer = HDBSCAN(min_cluster_size=100, gen_min_span_tree=True, prediction_data=True)
                clusterer.fit(X)
                ori_tree = clusterer.condensed_tree_
                tree = clusterer.condensed_tree_.to_pandas()
                link_tree = clusterer.single_linkage_tree_

                                
                # tree_path = f'{wo_save_root}/condensed_tree.pdf'
                # plt.rcParams['font.family'] = 'Times New Roman'
                # plt.figure(figsize=(15, 12))
                # if not os.path.exists(tree_path):
                #     clusterer.condensed_tree_.plot()    
                #     plt.savefig(tree_path) 

                max_node = tree.loc[tree['child_size'].idxmax()]
                node_id = max_node['child']
                
                def find_first_large_drop(tree, node_id, xi=0.02, depth=3):
                    """
                    Starting from a given node, traverse downward through the condensed tree to find the first cluster 
                    where the lambda value drops significantly — that is, the difference in lambda exceeds a given threshold. 
                    This point is considered a potential cluster split.

                    :param tree: A DataFrame representation of the condensed tree, where each row represents a clustering step.
                    :param node_id: The starting node ID from which to begin the traversal.
                    :param  stability threshold xi: The threshold for the difference in lambda to determine a significant drop.
                    :return: Information about the subtree (cluster) that satisfies the condition.
                    """
                    sub_trees_info = []
                    stack = [node_id] # stack is the "Q" in the original paper (line 6 in Algorithm 1 on page 8 )
                    iterations = 0
                    # max_iterations = 3
                    max_iterations = depth
                    while stack and iterations < max_iterations:
                        current_id = stack.pop()
                        current_node = tree[tree['child'] == current_id].iloc[0]
                        parent_id = current_node['parent']
                        # lambda_val is the density level of current_node 
                        lambda_val = current_node['lambda_val']
                        print(f'current_node:\n {current_node}')

                        children = tree.loc[tree['parent'] == current_id]
                        max_child = tree.loc[children['child_size'].idxmax()]
                        print(f"max child:\n {max_child}")
                        # max_child = tree.loc[children['child_size'].idxmax()]
                        # cal the  cluster stability
                        lambda_diff = max_child['lambda_val'] - lambda_val
                        print(f'lambda_diff: {lambda_diff}')
                        
                        if lambda_diff > xi:
                            return max_child, lambda_diff
                        
                        stack.append(max_child['child'])
                        iterations += 1

                    return None, None
                
            
                
               
                split_child, lambda_diff = find_first_large_drop(tree, node_id, xi=self.xi, depth=dp)

                print(f'**********************find the best wo_layer_num: {wo_layer_num}**********************')
                thres_lambda_val = split_child['lambda_val'] - 0.0001
                labels = link_tree.get_clusters(1/thres_lambda_val, min_cluster_size=100)
                final_labels = np.zeros(len(labels), dtype=int)
                # 计算每个聚类的大小并找出最大的聚类
                unique_labels, counts = np.unique(labels, return_counts=True)
                max_cluster_label = unique_labels[np.argmax(counts)]
                for label in unique_labels:
                    if label == -1:  # 噪音点已经设置为0
                        continue
                    elif label == max_cluster_label:
                        final_labels[labels == label] = 0
                    else:
                        final_labels[labels == label] = 1
                        
                y_preds = final_labels
                pred_clean_indices = torch.where(torch.from_numpy(y_preds) == 0)[0]
                # torch.save(clean_indices, f'{name_root}/pred_clean_indices.pt')
        
                print(f'y_preds: {y_preds}')
                print(f'y_true: {self.y_true}')
                tn, fp, fn, tp = metrics.confusion_matrix(self.y_true, y_preds).ravel()
                # tn, fp, fn, tp = metrics.confusion_matrix(y_true, labels).ravel()
                print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
                print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
                print("FNR: {:.2f}".format(fn / (tp + fn) * 100))        
                break
        
    # train model on the remain benign samples
    # def train_on_filter(self):
    #     args = self.args
    #     # indices_path = f"{supervisor.get_poison_set_dir(args)}/only_min/pred_indices/clean_model.pt"
    #     name_root = f'{supervisor.get_poison_set_dir(args)}/FLARE/'
    #     indices_path = f'{name_root}/clean_indices.pt'
    #     clean_indices = torch.load(indices_path).tolist()
    #     clean_trainset = Subset(self.trainset, clean_indices)
    #     self.train_model(self.args, clean_indices, model_path)
    
    
   