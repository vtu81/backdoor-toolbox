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
from utils.gradcam import GradCAM, GradCAMpp
from scipy.optimize import minimize
import math


class SentiNet(BackdoorDefense):
    """
    Assuming oracle knowledge of the used trigger.
    """
    
    name: str = 'sentinet'

    def __init__(self, args, defense_fpr: float = 0.05, N: int = 100):
        super().__init__(args)
        self.args = args
        
        # Only support localized attacks
        # support_list = ['adaptive_patch', 'badnet', 'badnet_all_to_all', 'dynamic', 'TaCT']
        # assert args.poison_type in support_list
        assert args.dataset in ['cifar10', 'gtsrb']

        self.defense_fpr = defense_fpr
        self.N = N
        
        self.folder_path = 'other_defenses_tool_box/results/Sentinet'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        
        self.random_img = self.normalizer(torch.rand((3, self.img_size, self.img_size))).cuda()

    def detect(self):
        args = self.args
        loader = generate_dataloader(dataset=self.dataset,
                                    dataset_path=config.data_dir,
                                    batch_size=1,
                                    split='valid',
                                    shuffle=True,
                                    drop_last=False)
        loader = tqdm(loader)
        
        clean_loader = generate_dataloader(dataset=self.dataset,
                                            dataset_path=config.data_dir,
                                            batch_size=100,
                                            split='test',
                                            shuffle=True,
                                            drop_last=False)
        clean_subset, val_subset, _ = torch.utils.data.random_split(clean_loader.dataset, [self.N, 400, len(clean_loader.dataset) - self.N - 400])
        clean_loader = torch.utils.data.DataLoader(clean_subset, batch_size=100, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
        
        
        est_fooled = []
        est_avgconf = []
        
        for i, (_input, _label) in enumerate(tqdm(val_loader)):
            _input, _label = _input.cuda(), _label.cuda()
            
            fooled_num = 0
            avgconf = 0
            
            model_gradcam = GradCAM(dict(type='resnet', arch=self.model.module, layer_name='layer4', input_size=(224, 224)), False)
            gradcam_mask, _ = model_gradcam(_input[0].unsqueeze(0))
            gradcam_mask = gradcam_mask.squeeze(0)
            v, _ = torch.topk(gradcam_mask.reshape(-1), k=int(len(gradcam_mask.reshape(-1)) * 0.15))
            gradcam_mask = (gradcam_mask > v[-1]).repeat([3, 1, 1])
            
            # from utils.gradcam_utils import visualize_cam
            # heatmap, result = visualize_cam(mask.cpu().detach(), self.denormalizer(_input[0]).cpu().detach())
            # torchvision.utils.save_image(result, "a.png")
            # torchvision.utils.save_image(self.denormalizer(_input[0]).cpu().detach(), "a0.png")
            # exit()
            
            
            for c_input, c_label in clean_loader:
                adv_input = c_input.clone().cuda()
                inert_input = c_input.clone().cuda()

                adv_input[:, gradcam_mask] = _input[:, gradcam_mask]
                inert_input[:, gradcam_mask] = self.normalizer(torch.rand_like(inert_input))[:, gradcam_mask].cuda()
                
                adv_output = self.model(adv_input)
                adv_pred = torch.argmax(adv_output, dim=1)
                fooled_num += torch.eq(adv_pred, _label).sum()
                
                inert_output = self.model(inert_input)
                inert_conf = torch.softmax(inert_output, dim=1)
                avgconf += inert_conf.max(dim=1)[0].sum()
            
            fooled = fooled_num / len(clean_loader.dataset)
            avgconf /= len(clean_loader.dataset)
            est_fooled.append(fooled.item())
            est_avgconf.append(avgconf.item())
            
        # torch.save(est_avgconf, os.path.join(poison_set_dir, f'SentiNet_est_avgconf_seed={args.seed}'))
        # torch.save(est_fooled, os.path.join(poison_set_dir, f'SentiNet_est_fooled_seed={args.seed}'))
    
    
        # Select the maximum marginal points by bins
        bin_size = 0.02
        x_min = np.min(np.array(est_avgconf))
        x_max = np.max(np.array(est_avgconf))
        n_bin = math.floor((x_max - x_min) / bin_size) + 1
        x = np.zeros(n_bin)
        y = np.zeros(n_bin)
        for i in range(len(est_avgconf)):
            avgconf = est_avgconf[i]
            fooled = est_fooled[i]
            k = math.floor((est_avgconf[i] - x_min) / bin_size)
            if y[k] <= fooled: x[k] = avgconf
            y[k] = max(y[k], fooled)
            
        for i in range(len(x)):
            x[i] = x_min + i * bin_size + bin_size / 2;
        
        # Fit a quadratic function for selected points
        from sklearn.preprocessing import PolynomialFeatures
        # est_avgconf = np.array(est_avgconf)
        # est_fooled = np.array(est_fooled)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        # poly_features = poly.fit_transform(est_avgconf.reshape(-1, 1))
        poly_features = poly.fit_transform(x.reshape(-1, 1))
        # print(poly_features.shape)
        
        from sklearn.linear_model import LinearRegression
        poly_reg_model = LinearRegression()
        # poly_reg_model.fit(poly_features, est_fooled)
        poly_reg_model.fit(poly_features, y)
        # print(poly_reg_model.coef_, poly_reg_model.intercept_)
        fit_func = lambda x: poly_reg_model.intercept_ + poly_reg_model.coef_[0] * x + poly_reg_model.coef_[1] * x ** 2
        
        # Estimate decision boundary
        d_thr = 0
        cnt = 0
        for i in range(len(est_avgconf)):
            x1 = est_avgconf[i]
            y1 = est_fooled[i]
            yp = poly_reg_model.intercept_ + poly_reg_model.coef_[0] * x1 + poly_reg_model.coef_[1] * x1 ** 2
            if yp > y1:
                loss_func = lambda x: (x - x1) ** 2 + (fit_func(x) - y1) ** 2
                res = minimize(loss_func, (2, 0), method='cobyla')
                d_thr += math.sqrt(res.fun)
                cnt += 1
        d_thr /= cnt
        
        # Determine y_plus
        x2 = 0
        y2 = fit_func(x2)
        x1 = 0
        y1 = y2+d_thr
        dt = 0;
        while dt < d_thr:
            y1 = y1 + 0.001
            loss_func = lambda x: (x - x1) ** 2 + (fit_func(x) - y1) ** 2
            res = minimize(loss_func, (2, 0), method='cobyla')
            dt = math.sqrt(res.fun)
        y_plus = y1 - y2
        # print("d_thr:", d_thr)
        # print("y_plus:", y_plus)
        thr_func = lambda x: poly_reg_model.intercept_ + y_plus + poly_reg_model.coef_[0] * x + poly_reg_model.coef_[1] * x ** 2
        
        
        plt.scatter(est_avgconf, est_fooled, marker='o', color='blue', s=5, alpha=1.0)
        plt.scatter(x, y, marker='o', color='green', s=5, alpha=1.0)
        x = np.linspace(x_min, x_max)
        y = fit_func(x)
        y_thr = thr_func(x)
        plt.plot(x, y, 'g', linewidth=3, label='fitted')
        plt.plot(x, y_thr, 'g', linestyle='dashed', linewidth=3, label='threshold')

        save_path = 'assets/SentiNet_est_%s.png' % (supervisor.get_dir_core(args, include_model_name=True))
        plt.xlabel("AvgConf")
        plt.ylabel("#Fooled")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(save_path)
        
        print("Saved figure at {}".format(save_path))
        plt.clf()
        

        
        clean_fooled = []
        clean_avgconf = []
        poison_fooled = []
        poison_avgconf = []
        
        for i, (_input, _label) in enumerate(loader):
            # if i > 30: break
            # For the clean input
            _input, _label = _input.cuda(), _label.cuda()
            fooled_num = 0
            avgconf = 0
            
            model_gradcam = GradCAM(dict(type='resnet', arch=self.model.module, layer_name='layer4', input_size=(224, 224)), False)
            gradcam_mask, _ = model_gradcam(_input[0].unsqueeze(0))
            gradcam_mask = gradcam_mask.squeeze(0)
            v, _ = torch.topk(gradcam_mask.reshape(-1), k=int(len(gradcam_mask.reshape(-1)) * 0.15))
            # gradcam_mask[gradcam_mask > v[-1]] = 1
            # gradcam_mask[gradcam_mask <= v[-1]] = 0
            gradcam_mask = (gradcam_mask > v[-1]).repeat([3, 1, 1])
            
            for c_input, c_label in clean_loader:
                adv_input = c_input.clone().cuda()
                inert_input = c_input.clone().cuda()
                
                adv_input[:, gradcam_mask] = _input[:, gradcam_mask]
                inert_input[:, gradcam_mask] = self.normalizer(torch.rand_like(inert_input))[:, gradcam_mask].cuda()
                
                adv_output = self.model(adv_input)
                adv_pred = torch.argmax(adv_output, dim=1)
                fooled_num += torch.eq(adv_pred, _label).sum()
                
                inert_output = self.model(inert_input)
                inert_conf = torch.softmax(inert_output, dim=1)
                # avgconf += torch.cat([inert_conf[x, y].unsqueeze(0) for x, y in list(zip(range(len(adv_pred)), adv_pred.tolist()))]).sum()
                avgconf += inert_conf.max(dim=1)[0].sum()
            
            fooled = fooled_num / len(clean_loader.dataset)
            avgconf /= len(clean_loader.dataset)
            # print(avgconf)
            clean_fooled.append(fooled.item())
            clean_avgconf.append(avgconf.item())
            
            # For the poison input
            poison_input, poison_label = self.poison_transform.transform(_input, _label)
            fooled_num = 0
            avgconf = 0
            for c_input, c_label in clean_loader:
                adv_input = c_input.clone().cuda()
                inert_input = c_input.clone().cuda()
                c_label = c_label.cuda()
                
                # Oracle (approximate) knowledge to the trigger position
                if args.poison_type == 'badnet' or args.poison_type == 'badnet_all_to_all':
                    dx = dy = 5
                    posx = self.img_size - dx
                    posy = self.img_size - dy
                    
                    adv_input[:, :, posx:posx+dx, posy:posy+dy] = poison_input[0, :, posx:posx+dx, posy:posy+dy]
                    inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.normalizer(torch.rand((inert_input.shape[0], 3, dx, dy))).cuda()
                    # inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.random_img[:, posx:posx+dx, posy:posy+dy]
                elif args.poison_type == 'TaCT' or args.poison_type == 'trojan':
                    dx = dy = 16
                    posx = self.img_size - dx
                    posy = self.img_size - dy
                    
                    adv_input[:, :, posx:posx+dx, posy:posy+dy] = poison_input[0, :, posx:posx+dx, posy:posy+dy]
                    inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.normalizer(torch.rand((inert_input.shape[0], 3, dx, dy))).cuda()
                    # inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.random_img[:, posx:posx+dx, posy:posy+dy]
                elif args.poison_type == 'dynamic' or args.poison_type == 'adaptive_patch':
                    trigger_mask = ((poison_input - _input).abs() > 1e-4)[0].cuda()
                    # self.debug_save_img(poison_input)
                    # print(trigger_mask.sum())
                    # print(poison_input.reshape(-1)[:10], _input.reshape(-1)[:10], trigger_mask.reshape(-1)[:10])
                    # exit()
                    adv_input[:, trigger_mask] = poison_input[0, trigger_mask]
                    # self.debug_save_img(adv_input[1])
                    # exit()
                    inert_input[:, trigger_mask] = self.normalizer(torch.rand(inert_input.shape))[:, trigger_mask].cuda()
                    # self.debug_save_img(inert_input[1])
                    # exit()
                else:
                    adv_input[:, gradcam_mask] = poison_input[:, gradcam_mask]
                    inert_input[:, gradcam_mask] = self.normalizer(torch.rand_like(inert_input))[:, gradcam_mask].cuda()
                
                adv_output = self.model(adv_input)
                adv_pred = torch.argmax(adv_output, dim=1)
                if args.poison_type != 'badnet_all_to_all':
                    fooled_num += torch.eq(adv_pred, poison_label).sum()
                else:
                    fooled_num += torch.eq(adv_pred, c_label + 1).sum()
                
                inert_output = self.model(inert_input)
                inert_conf = torch.softmax(inert_output, dim=1)
                # avgconf += torch.cat([inert_conf[x, y].unsqueeze(0) for x, y in list(zip(range(len(adv_pred)), adv_pred.tolist()))]).sum()
                avgconf += inert_conf.max(dim=1)[0].sum()

            fooled = fooled_num / len(clean_loader.dataset)
            avgconf /= len(clean_loader.dataset)
            poison_fooled.append(fooled.item())
            poison_avgconf.append(avgconf.item())

        plt.scatter(clean_avgconf, clean_fooled, marker='o', color='blue', s=5, alpha=1.0)
        plt.scatter(poison_avgconf, poison_fooled, marker='^', s=8, color='red', alpha=0.7)
        # x = np.linspace(x_min, x_max)
        # y = poly_reg_model.intercept_ + poly_reg_model.coef_[0] * x + poly_reg_model.coef_[1] * x ** 2
        plt.plot(x, y, 'g', linewidth=3, label='fitted')
        plt.plot(x, y_thr, 'g', linestyle='dashed', linewidth=3, label='threshold')
        save_path = 'assets/SentiNet_%s.png' % (supervisor.get_dir_core(args))
        plt.xlabel("AvgConf")
        plt.ylabel("#Fooled")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        print("Saved figure at {}".format(save_path))
        plt.savefig(save_path)
        
        clean_avgconf = torch.tensor(clean_avgconf)
        clean_fooled = torch.tensor(clean_fooled)
        poison_avgconf = torch.tensor(poison_avgconf)
        poison_fooled = torch.tensor(poison_fooled)
        all_avgconf = torch.zeros(len(poison_fooled) + len(clean_fooled))
        all_fooled = torch.zeros(len(poison_fooled) + len(clean_fooled))
        all_avgconf[:len(clean_avgconf)] = clean_avgconf
        all_fooled[:len(clean_fooled)] = clean_fooled
        all_avgconf[len(clean_avgconf):] = poison_avgconf
        all_fooled[len(clean_fooled):] = poison_fooled
        
        all_d = torch.zeros(len(poison_fooled) + len(clean_fooled))
        for i in tqdm(range(len(all_fooled))):
            x1 = all_avgconf[i].item()
            y1 = all_fooled[i].item()
            loss_func = lambda x: (x - x1) ** 2 + (fit_func(x) - y1) ** 2
            res = minimize(loss_func, (2, 0), method='cobyla')
            d1 = math.sqrt(res.fun)
            if y1 < fit_func(x1): d1 = -d1
            all_d[i] = d1
        
        # If a `defense_fpr` is explicitly specified, use it as the false positive rate to set the threshold, instead of the precomputed `d_thr`
        if self.defense_fpr is not None and args.poison_type != 'none':
            print("FPR is set to:", self.defense_fpr)
            clean_d = all_d[:len(clean_avgconf)]
            idx = math.ceil(self.defense_fpr * len(clean_d))
            d_thr = torch.sort(clean_d, descending=True)[0][idx] - 1e-8
        
        y_true = torch.zeros(len(poison_fooled) + len(clean_fooled))
        y_pred = torch.zeros(len(poison_fooled) + len(clean_fooled))
        y_true[len(clean_avgconf):] = 1
        y_pred = (all_d > d_thr).int().reshape(-1)
        
        print("f1_score:", metrics.f1_score(y_true, y_pred))
        print("precision_score:", metrics.precision_score(y_true, y_pred))
        print("recall_score (TPR):", metrics.recall_score(y_true, y_pred))
        print("accuracy_score:", metrics.accuracy_score(y_true, y_pred))
    
    
    def debug_save_img(self, t, path='a.png'):
        torchvision.utils.save_image(self.denormalizer(t.reshape(3, self.img_size, self.img_size)), path)