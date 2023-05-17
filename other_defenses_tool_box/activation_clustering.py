import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from . import BackdoorDefense
import config, os
from utils import supervisor
import torch, torchvision
from .tools import AverageMeter, generate_dataloader, tanh_func, to_numpy, jaccard_idx, normalize_mad, unpack_poisoned_train_set

"""
Adapted Activation Clustering as a backdoor input detector.
For comparison and reference only.
"""


def cluster_metrics(cluster_1, cluster_0):

    num = len(cluster_1) + len(cluster_0)
    features = torch.cat([cluster_1, cluster_0], dim=0)

    labels = torch.zeros(num)
    labels[:len(cluster_1)] = 1
    labels[len(cluster_1):] = 0

    ## Raw Silhouette Score
    raw_silhouette_score = silhouette_score(features, labels)
    return raw_silhouette_score


def get_features(data_loader, model, poison_transform, num_classes):

    model.eval()
    class_indices = [[] for _ in range(num_classes)]
    feats = []

    with torch.no_grad():
        sid = 0
        for i, (clean_data, clean_target) in enumerate(tqdm(data_loader)):
            clean_data, clean_target = clean_data.cuda(), clean_target.cuda()
            
            _, clean_feats = model(clean_data, return_hidden=True)
            this_batch_size = len(clean_target)
            for bid in range(this_batch_size):
                feats.append(clean_feats[bid])
                b_target = clean_target[bid].item()
                class_indices[b_target].append(sid + bid)
            sid += this_batch_size
            
        for i, (clean_data, clean_target) in enumerate(tqdm(data_loader)):
            clean_data, clean_target = clean_data.cuda(), clean_target.cuda()
            poison_data, poison_target = poison_transform.transform(clean_data, clean_target)
            
            _, poison_feats = model(poison_data, return_hidden=True)
            this_batch_size = len(clean_target)
            for bid in range(this_batch_size):
                feats.append(poison_feats[bid])
                b_target = poison_target[bid].item()
                class_indices[b_target].append(sid + bid)
            sid += this_batch_size
    return feats, class_indices


class AC(BackdoorDefense):
    name: str = 'AC'

    def __init__(self, args):
        super().__init__(args)
        self.args = args
    
    def detect(self, inspect_correct_predition_only=True, noisy_test=False):
        args = self.args
        
        from sklearn.cluster import KMeans
        
        test_set_loader = generate_dataloader(dataset=args.dataset, dataset_path=config.data_dir, split='test', data_transform=self.data_transform, shuffle=False, noisy_test=noisy_test)
        # loader = generate_dataloader(dataset=self.dataset, dataset_path=config.data_dir, batch_size=100, split='valid', shuffle=False, drop_last=False)

        suspicious_indices = []
        feats, class_indices = get_features(test_set_loader, self.model, self.poison_transform, self.num_classes)
        
        
        y_true = torch.cat((torch.zeros(len(test_set_loader.dataset)), torch.ones(len(test_set_loader.dataset))))
        y_score = torch.zeros_like(y_true)
        class_scores = []
        class_outliers = []
        for target_class in range(self.num_classes):
            
            # print('class - %d' % target_class)

            if len(class_indices[target_class]) <= 1: continue # no need to perform clustering...

            temp_feats = [feats[temp_idx].unsqueeze(dim=0) for temp_idx in class_indices[target_class]]
            temp_feats = torch.cat( temp_feats , dim=0)
            temp_feats = temp_feats - temp_feats.mean(dim=0)

            _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)

            axes = V[:, :10]
            projected_feats = torch.matmul(temp_feats, axes)
            projected_feats = projected_feats.cpu().numpy()

            # print('start k-means')
            kmeans = KMeans(n_clusters=2).fit(projected_feats)
            # print('end k-means')

            # by default, take the large cluster as the poisoned cluster (since all inference-time backdoor inputs are in the target class)
            if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
                clean_label = 0
            else:
                clean_label = 1


            score = silhouette_score(projected_feats, kmeans.labels_)

            outliers = []
            for (bool, idx) in zip((kmeans.labels_ != clean_label).tolist(), list(range(len(kmeans.labels_)))):
                y_score[class_indices[target_class][idx]] = torch.tensor(score) if bool else -torch.tensor(score)
                if bool:
                    outliers.append(class_indices[target_class][idx])

            class_scores.append(score)
            class_outliers.append(outliers)
            # print('[class-%d] silhouette_score = %f' % (target_class, score))

        print("Silhouette Score:", class_scores)
        suspicious_class = torch.tensor(class_scores).argmax()
        print("Suspicious Class:", suspicious_class)
        suspicious_indices = class_outliers[suspicious_class]

        y_pred = torch.zeros_like(y_true)
        y_pred[suspicious_indices] = 1
        
        
        
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
        
        
        
        from sklearn import metrics
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        print("")
        print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
        print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
        print("AUC: {:.4f}".format(auc))
        


def cleanser(inspection_set, model, num_classes, args, clusters=2):
    """
        adapted from : https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py
    """

    #from sklearn.decomposition import PCA
    #from sklearn.decomposition import FastICA
    from sklearn.cluster import KMeans

    kwargs = {'num_workers': 4, 'pin_memory': True}

    inspection_split_loader = torch.utils.data.DataLoader(
        inspection_set,
        batch_size=128, shuffle=False, **kwargs)

    suspicious_indices = []
    feats, class_indices = get_features(inspection_split_loader, model, num_classes)

    num_samples = len(inspection_set)



    if args.dataset == 'cifar10':
        threshold = 0.15
    elif args.dataset == 'gtsrb':
        threshold = 0.25
    elif args.dataset == 'imagenette':
        threshold = 0 # place holder, not used
    else:
        raise NotImplementedError('dataset %s is not supported' % args.datasets)

    for target_class in range(num_classes):

        print('class - %d' % target_class)

        if len(class_indices[target_class]) <= 1: continue # no need to perform clustering...

        temp_feats = [feats[temp_idx].unsqueeze(dim=0) for temp_idx in class_indices[target_class]]
        temp_feats = torch.cat( temp_feats , dim=0)
        temp_feats = temp_feats - temp_feats.mean(dim=0)

        _, _, V = torch.svd(temp_feats, compute_uv=True, some=False)

        axes = V[:, :10]
        projected_feats = torch.matmul(temp_feats, axes)
        projected_feats = projected_feats.cpu().numpy()

        print(projected_feats.shape)

        #projector = PCA(n_components=10)
        #print('start pca')
        #projected_feats = projector.fit_transform(temp_feats)
        #print('end pca')

        print('start k-means')
        kmeans = KMeans(n_clusters=2).fit(projected_feats)
        print('end k-means')

        # by default, take the large cluster as the poisoned cluster (since all inference-time backdoor inputs are in the target class)
        if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
            clean_label = 0
        else:
            clean_label = 1

        outliers = []
        for (bool, idx) in zip((kmeans.labels_ != clean_label).tolist(), list(range(len(kmeans.labels_)))):
            if bool:
                outliers.append(class_indices[target_class][idx])

        score = silhouette_score(projected_feats, kmeans.labels_)
        print('[class-%d] silhouette_score = %f' % (target_class, score))
        # if score > threshold:# and len(outliers) < len(kmeans.labels_) * 0.35:
        
        if len(outliers) > len(kmeans.labels_) * 0.65: # if one of the two clusters is abnormally large
            print(f"Outlier Num in Class {target_class}:", len(outliers))
            suspicious_indices += outliers

    return suspicious_indices