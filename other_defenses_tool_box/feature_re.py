import os
import pdb
import random
import torch.nn as nn
from utils.unet_model import UNet
import torch
import config
from torchvision import transforms
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from torch.utils.data import Subset, DataLoader
from utils.tools import test
import numpy as np
from functools import reduce


class FeatureRE(BackdoorDefense):
    name: str = 'FeatureRE'

    def __init__(self, args, wp_epochs=100, epochs=400):
        super().__init__(args)
        self.args = args
        self.wp_epochs = wp_epochs
        self.epchs = epochs
        # test set --- clean
        # std_test - > 10000 full, val -> 2000 (for detection), test -> 8000 (for accuracy)
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='test',
                                               shuffle=False,
                                               drop_last=False,
                                               )

        self.train_loader = generate_dataloader(dataset=self.dataset,
                                                dataset_path=config.data_dir,
                                                batch_size=100,
                                                split='train',
                                                shuffle=False,
                                                drop_last=False,
                                                )
        self.train_set = self.train_loader.dataset
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.re_dataloader = self.get_dataloader_label_remove()

        self.classifier = self.model
        self.classifier.cuda()
        self.AE = UNet(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4)
        self.AE.cuda()
        self.AE.train()
        self._EPSILON = 1e-7
        self.init_mask, self.feature_shape = self.get_mask()
        self.mask_tanh = nn.Parameter(torch.tensor(self.init_mask))
        self.all_features, self.weight_map_class = self.get_range()

    def detect(self):
        weight_p = 1
        weight_acc = 1
        weight_std = 1

        optimizerR = torch.optim.Adam(self.AE.parameters(), lr=0.001, betas=(0.5, 0.9))
        optimizerR_mask = torch.optim.Adam([self.mask_tanh], lr=1e-1, betas=(0.5, 0.9))
        self.AE.train()

        mixed_value_best = float("inf")
        # Learning the transformation
        for epoch in range(self.epchs):
            total_pred = 0
            true_pred = 0
            loss_ce_list = []
            loss_dist_list = []
            loss_list = []
            acc_list = []
            p_loss_list = []
            loss_mask_norm_list = []
            loss_std_list = []

            for batch_idx, (inputs, labels) in enumerate(self.re_dataloader):
                self.AE.train()
                self.mask_tanh.requires_grad = False
                optimizerR.zero_grad()
                inputs = inputs.cuda()
                sample_num = inputs.shape[0]
                total_pred += sample_num
                target_labels = torch.ones(sample_num, dtype=torch.int64).cuda() * self.target_class
                if epoch < self.wp_epochs:
                    predictions, features, x_before_ae, x_after_ae = self.forward_ae(inputs)
                else:
                    predictions, features, x_before_ae, x_after_ae, features_ori = self.forward_ae_mask_p(inputs)
                loss_ce = self.criterion(predictions, target_labels)
                mse_loss = torch.nn.MSELoss(size_average=True).cuda()(x_after_ae, x_before_ae)
                if epoch < self.wp_epochs:
                    dist_loss = torch.cosine_similarity(self.weight_map_class[self.target_class].reshape(-1),
                                                        features.mean(0).reshape(-1), dim=0)
                else:
                    dist_loss = torch.cosine_similarity(self.weight_map_class[self.target_class].reshape(-1),
                                                        features_ori.mean(0).reshape(-1), dim=0)
                acc_list_ = []
                minibatch_accuracy_ = torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach() / sample_num
                acc_list_.append(minibatch_accuracy_)
                acc_list_ = torch.stack(acc_list_)
                avg_acc_G = torch.mean(acc_list_)
                acc_list.append(minibatch_accuracy_)
                p_loss = mse_loss
                p_loss_bound = 0.15
                loss_std_bound = 1.0
                atk_succ_threshold = 0.9
                if epoch < self.wp_epochs:
                    if p_loss > p_loss_bound:
                        total_loss = loss_ce + p_loss * 100
                    else:
                        total_loss = loss_ce
                else:
                    loss_std = (features_ori * self.get_raw_mask()).std(0).sum()
                    loss_std = loss_std / (torch.norm(self.get_raw_mask(), 1))
                    total_loss = dist_loss * 5
                    if dist_loss < 0:
                        total_loss = total_loss - dist_loss * 5
                    if loss_std > loss_std_bound:
                        total_loss = total_loss + loss_std * 10 * (1 + weight_std)
                    if p_loss > p_loss_bound:
                        total_loss = total_loss + p_loss * 10 * (1 + weight_p)
                    if avg_acc_G.item() < atk_succ_threshold:
                        total_loss = total_loss + 1 * loss_ce * (1 + weight_acc)
                total_loss.backward()
                optimizerR.step()
                mask_norm_bound = int(reduce(lambda x, y: x * y, self.feature_shape) * 0.03)

                if epoch >= self.wp_epochs:
                    for k in range(1):
                        self.AE.eval()
                        self.mask_tanh.requires_grad = True
                        optimizerR_mask.zero_grad()
                        predictions, features, x_before_ae, x_after_ae, features_ori = self.forward_ae_mask_p(inputs)
                        loss_mask_ce = self.criterion(predictions, target_labels)
                        loss_mask_norm = torch.norm(self.get_raw_mask(), 1)
                        loss_mask_total = loss_mask_ce
                        if loss_mask_norm > mask_norm_bound:
                            loss_mask_total = loss_mask_total + loss_mask_norm
                        loss_mask_total.backward()
                        optimizerR_mask.step()

                loss_ce_list.append(loss_ce.detach())
                loss_dist_list.append(dist_loss.detach())
                loss_list.append(total_loss.detach())
                true_pred += torch.sum(torch.argmax(predictions, dim=1) == target_labels).detach()

                if epoch >= self.wp_epochs:
                    p_loss_list.append(p_loss)
                    loss_mask_norm_list.append(loss_mask_norm)
                    loss_std_list.append(loss_std)

            loss_ce_list = torch.stack(loss_ce_list)
            loss_dist_list = torch.stack(loss_dist_list)
            loss_list = torch.stack(loss_list)
            acc_list = torch.stack(acc_list)

            avg_loss_ce = torch.mean(loss_ce_list)
            avg_loss_dist = torch.mean(loss_dist_list)
            avg_loss = torch.mean(loss_list)
            avg_acc = torch.mean(acc_list)

            if epoch >= self.wp_epochs:
                p_loss_list = torch.stack(p_loss_list)
                loss_mask_norm_list = torch.stack(loss_mask_norm_list)
                loss_std_list = torch.stack(loss_std_list)

                avg_p_loss = torch.mean(p_loss_list)
                avg_loss_mask_norm = torch.mean(loss_mask_norm_list)
                avg_loss_std = torch.mean(loss_std_list)
                print("avg_ce_loss:", avg_loss_ce)
                print("avg_asr:", avg_acc)
                print("avg_p_loss:", avg_p_loss)
                print("avg_loss_mask_norm:", avg_loss_mask_norm)
                print("avg_loss_std:", avg_loss_std)

                if avg_acc.item() < atk_succ_threshold:
                    print("@avg_asr lower than bound")
                if avg_p_loss > 1.0 * p_loss_bound:
                    print("@avg_p_loss larger than bound")
                if avg_loss_mask_norm > 1.0 * mask_norm_bound:
                    print("@avg_loss_mask_norm larger than bound")
                if avg_loss_std > 1.0 * loss_std_bound:
                    print("@avg_loss_std larger than bound")

                mixed_value = avg_loss_dist.detach() - avg_acc + max(avg_p_loss.detach() - p_loss_bound,
                                                                     0) / p_loss_bound + max(
                    avg_loss_mask_norm.detach() - mask_norm_bound, 0) / mask_norm_bound + max(
                    avg_loss_std.detach() - loss_std_bound, 0) / loss_std_bound
                print("mixed_value:", mixed_value)
                if mixed_value < mixed_value_best:
                    mixed_value_best = mixed_value
                weight_p = max(avg_p_loss.detach() - p_loss_bound, 0) / p_loss_bound
                weight_acc = max(atk_succ_threshold - avg_acc, 0) / atk_succ_threshold
                weight_std = max(avg_loss_std.detach() - loss_std_bound, 0) / loss_std_bound

            print(
                "  Result: ASR: {:.3f} | Cross Entropy Loss: {:.6f} | Dist Loss: {:.6f} | Mixed_value best: {:.6f}".format(
                    true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_dist, mixed_value_best
                )
            )

    def get_dataloader_label_remove(self):
        idx = []
        dataloader_total = torch.utils.data.DataLoader(self.train_set, batch_size=1, pin_memory=True, shuffle=False)
        for batch_idx, (inputs, targets) in enumerate(dataloader_total):
            if targets.item() != self.target_class:
                idx.append(batch_idx)

        class_dataset = torch.utils.data.Subset(self.train_set, idx)
        dataloader_class = torch.utils.data.DataLoader(class_dataset, batch_size=100, pin_memory=True, shuffle=True)

        return dataloader_class

    def get_mask(self):
        with torch.no_grad():
            feature_shape = []
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                features = self.classifier.from_input_to_features(inputs.cuda())
                for i in range(1, len(features.shape)):
                    feature_shape.append(features.shape[i])
                break
            return torch.ones(feature_shape), feature_shape

    def get_range(self):
        with torch.no_grad():
            test_dataloader = self.train_loader
            features_list = []
            features_list_class = [[] for i in range(self.num_classes)]
            for batch_idx, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.cuda()
                features = self.classifier.from_input_to_features(inputs)
                features_list.append(features)
                for i in range(inputs.shape[0]):
                    features_list_class[labels[i].item()].append(features[i].unsqueeze(0))
            all_features = torch.cat(features_list, dim=0)
            weight_map_class = []
            for i in range(self.num_classes):
                feature_mean_class = torch.cat(features_list_class[i], dim=0).mean(0)
                weight_map_class.append(feature_mean_class)

            return all_features, weight_map_class

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        bounded = mask / (2 + self._EPSILON) + 0.5
        return bounded

    def forward_ae(self, x):
        x_before_ae = x
        x = self.AE(x)
        x_after_ae = x
        features = self.classifier.from_input_to_features(x)
        out = self.classifier.from_features_to_output(features)
        return out, features, x_before_ae, x_after_ae

    def forward_ae_mask_p(self, x):
        mask = self.get_raw_mask()
        x_before_ae = x
        x = self.AE(x)
        x_after_ae = x
        features = self.classifier.from_input_to_features(x)
        reference_features_index_list = np.random.choice(range(self.all_features.shape[0]), features.shape[0],
                                                         replace=True)
        reference_features = self.all_features[reference_features_index_list]
        features_ori = features
        features = mask * features + (1 - mask) * reference_features.reshape(features.shape)
        out = self.classifier.from_features_to_output(features)

        return out, features, x_before_ae, x_after_ae, features_ori
