import random
import numpy as np
import torch
import os
from torchvision import transforms
import argparse
from torch import nn
from utils import supervisor, tools, default_args
import config
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import silhouette_score

parser = argparse.ArgumentParser()
parser.add_argument('-method', type=str, required=False, default='pca',
                    choices=['pca', 'tsne', 'oracle', 'mean_diff', 'SS'])
parser.add_argument('-dataset', type=str, required=False, default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str,  required=True,
        choices=default_args.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False, default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float,  required=False, default=None)
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-model', type=str, required=False, default=None)
parser.add_argument('-model_path', required=False, default=None)
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-target_class', type=int, default=-1)
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
tools.setup_seed(args.seed)

if args.target_class == -1:
    target_class = config.target_class[args.dataset]
else:
    target_class = args.target_class

if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]

batch_size = 128
kwargs = {'num_workers': 4, 'pin_memory': True}


class mean_diff_visualizer:

    def fit_transform(self, clean, poison):
        clean_mean = clean.mean(dim=0)
        poison_mean = poison.mean(dim=0)
        mean_diff = poison_mean - clean_mean
        print("Mean L2 distance between poison and clean:", torch.norm(mean_diff, p=2).item())

        proj_clean_mean = torch.matmul(clean, mean_diff)
        proj_poison_mean = torch.matmul(poison, mean_diff)

        return proj_clean_mean, proj_poison_mean


class oracle_visualizer:

    def __init__(self):
        self.clf = svm.LinearSVC()

    def fit_transform(self, clean, poison):

        clean = clean.numpy()
        num_clean = len(clean)

        poison = poison.numpy()
        num_poison = len(poison)

        # print(clean.shape, poison.shape)

        X = np.concatenate([clean, poison], axis=0)
        y = []

        for _ in range(num_clean):
            y.append(0)
        for _ in range(num_poison):
            y.append(1)

        self.clf.fit(X, y)
        print("SVM Accuracy:", self.clf.score(X, y))

        norm = np.linalg.norm(self.clf.coef_)
        self.clf.coef_ = self.clf.coef_ / norm
        self.clf.intercept_ = self.clf.intercept_ / norm

        projection = self.clf.decision_function(X)

        return projection[:num_clean], projection[num_clean:]


class spectral_visualizer:

    def fit_transform(self, clean, poison):
        all_features = torch.cat((clean, poison), dim=0)
        all_features -= all_features.mean(dim=0)
        _, _, V = torch.svd(all_features, compute_uv=True, some=False)
        vec = V[:, 0]  # the top right singular vector is the first column of V
        vals = []
        for j in range(all_features.shape[0]):
            vals.append(torch.dot(all_features[j], vec).pow(2))
        vals = torch.tensor(vals)

        print(vals.shape)

        return vals[:clean.shape[0]], vals[clean.shape[0]:]



if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'gtsrb':
    num_classes = 43
elif args.dataset == 'imagenette':
    num_classes = 10
else:
    raise NotImplementedError('<Unimplemented Dataset> %s' % args.dataset)

data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)


arch = supervisor.get_arch(args)
# Set up Poisoned Set
poison_set_dir = supervisor.get_poison_set_dir(args)
if os.path.exists(os.path.join(poison_set_dir, 'data')): # if old version
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
if os.path.exists(os.path.join(poison_set_dir, 'imgs')): # if new version
    poisoned_set_img_dir = os.path.join(poison_set_dir, 'imgs')
poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')

poisoned_set = tools.IMG_Dataset(data_dir=poisoned_set_img_dir,
                                    label_path=poisoned_set_label_path, transforms=data_transform)

poisoned_set_loader = torch.utils.data.DataLoader(
        poisoned_set,
        batch_size=batch_size, shuffle=False, **kwargs)

poison_indices = torch.tensor(torch.load(poison_indices_path))


test_set_dir = 'clean_set/%s/test_split/' % args.dataset
test_set_img_dir = os.path.join(test_set_dir, 'data')
test_set_label_path = os.path.join(test_set_dir, 'labels')
test_set = tools.IMG_Dataset(data_dir=test_set_img_dir, label_path=test_set_label_path,
                             transforms=data_transform)
test_set_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size, shuffle=False, **kwargs
)





model_list = []
alias_list = []


"""
if args.poison_type == 'none':  # no poison => load vanilla data and model
    path = os.path.join('models', '%s_vanilla_no_aug.pt' % args.dataset)
    model_list.append(path)
    alias_list.append('vanilla_no_aug')

    path = os.path.join('models', '%s_vanilla_aug.pt' % args.dataset)
    model_list.append(path)
    alias_list.append('vanilla_aug')"""

if (hasattr(args, 'model_path') and args.model_path is not None) or (hasattr(args, 'model') and args.model is not None):
    path = supervisor.get_model_dir(args)
    model_list.append(path)
    alias_list.append('assigned')

else:
    # args.no_aug = True
    # #path = os.path.join(poison_set_dir, 'full_base_no_aug.pt') #
    # path = supervisor.get_model_dir(args)
    # model_list.append(path)
    # alias_list.append(supervisor.get_model_name(args))

    args.no_aug = False
    #path = os.path.join(poison_set_dir, 'full_base_aug.pt') #supervisor.get_model_dir(args)
    path = supervisor.get_model_dir(args)
    model_list.append(path)
    alias_list.append(supervisor.get_model_name(args))


poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                       target_class=target_class,
                                                       trigger_transform=data_transform,
                                                       is_normalized_input=True,
                                                       alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                       trigger_name=args.trigger, args=args)


if args.poison_type == 'TaCT':
    source_classes = [config.source_class]
else:
    source_classes = None


for vid, path in enumerate(model_list):

    ckpt = torch.load(path)

    # base model for poison detection
    model = arch(num_classes=num_classes)
    model.load_state_dict(ckpt)
    model = nn.DataParallel(model)
    model = model.cuda()
    model.eval()


    # Begin Visualization
    print("Visualizing model '{}' on {}...".format(path, args.dataset))

    print('[test]')
    tools.test(model, test_set_loader, poison_test=True, poison_transform=poison_transform, num_classes=num_classes, source_classes=source_classes)



    targets = []
    features = []
    clean_features = []
    poisoned_features = []


    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(poisoned_set_loader):
            data, target = data.cuda(), target.cuda()  # train set batch
            targets.append(target)
            _, feature = model.forward(data, return_hidden=True)
            features.append(feature.cpu().detach())

    targets = torch.cat(targets, dim=0)
    targets = targets.cpu()
    features = torch.cat(features, dim=0)
    ids = torch.tensor(list(range(len(poisoned_set))))

    if len(poison_indices) == 0:

        if args.method == 'pca':
            visualizer = PCA(n_components=2)
        elif args.method == 'tsne':
            visualizer = TSNE(n_components=2)
        else:
            raise NotImplementedError('Visualization Method %s is Not Implemented!' % args.method)

        non_poison_indices = list(set(list(range(len(poisoned_set)))) - set(poison_indices.tolist()))
        #print(non_poison_indices)
        clean_targets = targets[non_poison_indices]
        print("Total Clean:", len(clean_targets))
        print("Total Poisoned:", 0)

        clean_features = features[non_poison_indices]

        class_clean_features = clean_features[clean_targets == target_class]
        clean_ids = ids[non_poison_indices]
        class_clean_ids = clean_ids[clean_targets == target_class]

        reduced_features = visualizer.fit_transform(
            class_clean_features)  # all features vector under the label i

        #plt.scatter(reduced_features[:, 0], reduced_features[:, 1], facecolors='none', marker='o',
        #            color='blue', label='clean')

        plt.scatter(reduced_features[:, 0], reduced_features[:, 1], marker='o', color='blue', s=5, alpha=0.5)
        plt.axis('off')
        save_path = 'assets/%s_%s_%s_class=%d.png' % (args.method, supervisor.get_dir_core(args, include_poison_seed=True), alias_list[vid], target_class)
        plt.savefig(save_path)
        print("Saved figure at {}".format(save_path))
        plt.clf()

    else:


        non_poison_indices = list(set(list(range(len(poisoned_set)))) - set(poison_indices.tolist()))

        clean_targets = targets[non_poison_indices]
        poisoned_targets = targets[poison_indices]

        print("Total Clean:", len(clean_targets))
        print("Total Poisoned:", len(poisoned_targets))

        clean_features = features[non_poison_indices]
        poisoned_features = features[poison_indices]

        clean_ids = ids[non_poison_indices]
        poisoned_ids = ids[poison_indices]

        class_clean_features = clean_features[clean_targets == target_class]
        class_poisoned_features = poisoned_features[poisoned_targets == target_class]
        class_clean_ids = clean_ids[clean_targets == target_class]
        class_poisoned_ids = poisoned_ids[poisoned_targets == target_class]

        num_clean = len(class_clean_features)
        num_poisoned = len(class_poisoned_features)

        feats = torch.cat([class_clean_features, class_poisoned_features], dim=0)
        ids = list(range(0,len(feats)))
        random.shuffle(ids)
        #class_clean_features = feats[ids[:num_clean]]
        #class_poisoned_features = feats[ids[-num_poisoned:]]
        # class_poisoned_features = poisoned_features


        class_clean_mean = class_clean_features.mean(dim=0)
        print(class_clean_mean.shape)
        clean_dis = torch.norm(class_clean_features - class_clean_mean, dim=1).mean()
        poison_dis = torch.norm(class_poisoned_features - class_clean_mean, dim=1).mean()
        print('clean_dis: %f, poison_dis: %f' % (clean_dis, poison_dis))

        tmp_labels = [0] * len(class_clean_features) + [1] * len(class_poisoned_features)
        silhouette = silhouette_score(feats, tmp_labels)
        print('Silhouette Score:', silhouette)
        # exit()

        if args.method == 'pca':
            visualizer = PCA(n_components=2)
        elif args.method == 'tsne':
            visualizer = TSNE(n_components=2)
        elif args.method == 'oracle':
            visualizer = oracle_visualizer()
        elif args.method == 'mean_diff':
            visualizer = mean_diff_visualizer()
        elif args.method == 'SS':
            visualizer = spectral_visualizer()
        else:
            raise NotImplementedError('Visualization Method %s is Not Implemented!' % args.method)


        if args.method == 'oracle':
            clean_projection, poison_projection = visualizer.fit_transform(class_clean_features,
                                                                           class_poisoned_features)
            # print(clean_projection)
            # print(poison_projection)

            # bins = np.linspace(-2, 2, 100)
            plt.figure(figsize=(7, 5))
            # plt.xlim([-3, 3])
            plt.ylim([0, 100])

            plt.hist(clean_projection, bins='doane', color='blue', alpha=0.5, label='Clean', edgecolor='black')
            plt.hist(poison_projection, bins='doane', color='red', alpha=0.5, label='Poison', edgecolor='black')
            
            # plt.xlabel("Distance")
            # plt.ylabel("Number")
            # plt.axis('off')
            # plt.legend()
        elif args.method == 'mean_diff':
            clean_projection, poison_projection = visualizer.fit_transform(class_clean_features, class_poisoned_features)
            # all_projection = torch.cat((clean_projection, poison_projection), dim=0)

            # bins = np.linspace(-5, 5, 50)
            plt.figure(figsize=(7, 5))

            # plt.hist(all_projection.cpu().detach().numpy(), bins='doane', alpha=1, label='all', linestyle='dashed', color='black', histtype="step", edgecolor='black')
            plt.hist(clean_projection.cpu().detach().numpy(), color='blue', bins='doane', alpha=0.5, label='Clean', edgecolor='black')
            plt.hist(poison_projection.cpu().detach().numpy(), color='red', bins='doane', alpha=0.5, label='Poison', edgecolor='black')
            
            plt.xlabel("Distance")
            plt.ylabel("Number")
            plt.legend()
        elif args.method == 'SS':
            clean_projection, poison_projection = visualizer.fit_transform(class_clean_features, class_poisoned_features)
            # all_projection = torch.cat((clean_projection, poison_projection), dim=0)

            # bins = np.linspace(-5, 5, 50)
            plt.figure(figsize=(7, 5))
            plt.ylim([0, 300])

            # plt.hist(all_projection.cpu().detach().numpy(), bins='doane', alpha=1, label='all', linestyle='dashed', color='black', histtype="step", edgecolor='black')
            plt.hist(clean_projection.cpu().detach().numpy(), color='blue', bins='doane', alpha=0.5, label='Clean', edgecolor='black')
            plt.hist(poison_projection.cpu().detach().numpy(), color='red', bins=20, alpha=0.5, label='Poison', edgecolor='black')
            
            plt.xlabel("Distance")
            plt.ylabel("Number")
            plt.legend()
        else:
            reduced_features = visualizer.fit_transform( torch.cat([class_clean_features, class_poisoned_features], dim=0) ) # all features vector under the label

            plt.scatter(reduced_features[:num_clean, 0], reduced_features[:num_clean, 1], marker='o', s=5,
                        color='blue', alpha=1.0)
            plt.scatter(reduced_features[num_clean:, 0], reduced_features[num_clean:, 1], marker='^', s=8,
                        color='red', alpha=0.7)


            plt.axis('off')


        save_path = 'assets/%s_%s_%s_class=%d.png' % (args.method, supervisor.get_dir_core(args, include_poison_seed=True), alias_list[vid], target_class)
        plt.tight_layout()
        plt.savefig(save_path)
        print("Saved figure at {}".format(save_path))

        plt.clf()