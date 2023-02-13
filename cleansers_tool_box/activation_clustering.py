import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import silhouette_score

def cluster_metrics(cluster_1, cluster_0):

    num = len(cluster_1) + len(cluster_0)
    features = torch.cat([cluster_1, cluster_0], dim=0)

    labels = torch.zeros(num)
    labels[:len(cluster_1)] = 1
    labels[len(cluster_1):] = 0

    ## Raw Silhouette Score
    raw_silhouette_score = silhouette_score(features, labels)
    return raw_silhouette_score



def get_features(data_loader, model, num_classes):

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
                feats.append(x_feats[bid])
                b_target = ins_target[bid].item()
                class_indices[b_target].append(sid + bid)
            sid += this_batch_size
    return feats, class_indices


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

        # by default, take the smaller cluster as the poisoned cluster
        if kmeans.labels_.sum() >= len(kmeans.labels_) / 2.:
            clean_label = 1
        else:
            clean_label = 0

        outliers = []
        for (bool, idx) in zip((kmeans.labels_ != clean_label).tolist(), list(range(len(kmeans.labels_)))):
            if bool:
                outliers.append(class_indices[target_class][idx])

        score = silhouette_score(projected_feats, kmeans.labels_)
        print('[class-%d] silhouette_score = %f' % (target_class, score))
        # if score > threshold:# and len(outliers) < len(kmeans.labels_) * 0.35:
        if len(outliers) < len(kmeans.labels_) * 0.35: # if one of the two clusters is abnormally small
            print(f"Outlier Num in Class {target_class}:", len(outliers))
            suspicious_indices += outliers

    return suspicious_indices