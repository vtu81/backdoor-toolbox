import torch, torchvision
import numpy as np
from tqdm import tqdm
from other_defenses_tool_box.tools import generate_dataloader
import random, os
import config
import matplotlib.pyplot as plt
from utils.tools import unpack_poisoned_train_set
from utils import supervisor
from utils.gradcam import GradCAM, GradCAMpp
from torchvision import transforms
import math
from scipy.optimize import minimize
import time

class SentiNet():
    """
    Assuming oracle knowledge of the used trigger.
    """
    
    name: str = 'sentinet'

    def __init__(self, args, model, defense_fpr: float = None, N: int = 100):
        self.args = args
        
        # Only support localized attacks
        # support_list = ['none', 'adaptive_patch', 'badnet', 'trojan', 'badnet_all_to_all', 'dynamic', 'TaCT']
        # assert args.poison_type in support_list
        assert args.dataset in ['cifar10', 'gtsrb']

        self.defense_fpr = defense_fpr
        self.model = model
        self.N = N
        
        if args.dataset == 'cifar10':
            self.normalizer = transforms.Compose([
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
            ])
            self.denormalizer = transforms.Compose([
                transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
            ])
            self.img_size = 32
        elif args.dataset == 'gtsrb':
            self.normalizer = transforms.Compose([
                transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
            ])
            self.denormalizer = transforms.Compose([
                transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
                                        (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
            ])
            self.img_size = 32
        else: raise NotImplementedError()
            
        self.random_img = self.normalizer(torch.rand((3, self.img_size, self.img_size))).cuda()

    def cleanse(self):
        args = self.args
        start_time = time.perf_counter()
        
        # Poisoned train set
        poison_set_dir, poisoned_set_loader, poison_indices, _ = unpack_poisoned_train_set(args, shuffle=False, batch_size=1)
        clean_indices = list(set(list(range(len(poisoned_set_loader.dataset)))) - set(poison_indices))
        
        # Original clean train set
        poison_type = args.poison_type
        poison_rate = args.poison_rate
        args.poison_type = 'none'
        args.poison_rate = 0
        _, original_set_loader, _, _ = unpack_poisoned_train_set(args, shuffle=False, batch_size=1)
        original_set = original_set_loader.dataset
        args.poison_type = poison_type
        args.poison_rate = poison_rate

        
        # `val_loader` is used to estimate the decision boundary
        val_loader = generate_dataloader(dataset=self.args.dataset,
                                            dataset_path=config.data_dir,
                                            batch_size=100,
                                            split='val',
                                            shuffle=False,
                                            drop_last=False)
        val_subset, _ = torch.utils.data.random_split(val_loader.dataset, [400, len(val_loader.dataset) - 400])
        val_loader = torch.utils.data.DataLoader(val_subset, batch_size=1, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
        
        # `clean_loader` provides the samples to add patches on
        clean_loader = generate_dataloader(dataset=self.args.dataset,
                                            dataset_path=config.data_dir,
                                            batch_size=100,
                                            split='test',
                                            shuffle=False,
                                            drop_last=False)
        clean_subset, _ = torch.utils.data.random_split(clean_loader.dataset, [self.N, len(clean_loader.dataset) - self.N])
        clean_loader = torch.utils.data.DataLoader(clean_subset, batch_size=100, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
        
        
        # First estimate the decision boundary 
        # if os.path.exists(os.path.join(poison_set_dir, f'SentiNet_est_avgconf_seed={args.seed}')) and os.path.exists(os.path.join(poison_set_dir, f'SentiNet_est_fooled_seed={args.seed}')):
        if False:
            est_avgconf = torch.load(os.path.join(poison_set_dir, f'SentiNet_est_avgconf_seed={args.seed}'))
            est_fooled = torch.load(os.path.join(poison_set_dir, f'SentiNet_est_fooled_seed={args.seed}'))
        else:
            est_fooled = []
            est_avgconf = []
            
            for i, (_input, _label) in enumerate(tqdm(val_loader)):
                _input, _label = _input.cuda(), _label.cuda()
                
                fooled_num = 0
                avgconf = 0
                
                # # simulate GradCAM map with a randomized central square area for clean inputs
                # from random import random
                # from numpy.random import normal
                # if args.dataset == 'gtsrb':
                #     scale = random() * 4 + 2
                # elif args.dataset == 'cifar10':
                #     scale = random() * 4 + 2
                # else: raise NotImplementedError()
                # # scale = torch.normal(mean=4.0, std=0.5, size=(1,)).clamp(2, 6).item()
                # # scale = 6
                
                model_gradcam = GradCAM(dict(type='resnet', arch=self.model.module, layer_name='layer4', input_size=(224, 224)), False)
                gradcam_mask, _ = model_gradcam(_input[0].unsqueeze(0))
                gradcam_mask = gradcam_mask.squeeze(0)
                v, _ = torch.topk(gradcam_mask.reshape(-1), k=int(len(gradcam_mask.reshape(-1)) * 0.15))
                # gradcam_mask[gradcam_mask > v[-1]] = 1
                # gradcam_mask[gradcam_mask <= v[-1]] = 0
                gradcam_mask = (gradcam_mask > v[-1]).repeat([3, 1, 1])
                
                # from utils.gradcam_utils import visualize_cam
                # heatmap, result = visualize_cam(mask.cpu().detach(), self.denormalizer(_input[0]).cpu().detach())
                # torchvision.utils.save_image(result, "a.png")
                # torchvision.utils.save_image(self.denormalizer(_input[0]).cpu().detach(), "a0.png")
                # exit()
                
                
                for c_input, c_label in clean_loader:
                    adv_input = c_input.clone().cuda()
                    inert_input = c_input.clone().cuda()
                    
                    # st_cd = int(self.img_size / scale)
                    # ed_cd = self.img_size - st_cd
                    
                    # adv_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = _input[0, :, st_cd:ed_cd, st_cd:ed_cd]
                    # inert_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = self.normalizer(torch.rand((inert_input.shape[0], 3, ed_cd - st_cd, ed_cd - st_cd))).cuda()
                    
                    adv_input[:, gradcam_mask] = _input[:, gradcam_mask]
                    inert_input[:, gradcam_mask] = self.normalizer(torch.rand_like(inert_input))[:, gradcam_mask].cuda()
                    
                    # torchvision.utils.save_image(self.denormalizer(adv_input), "a.png")
                    # torchvision.utils.save_image(self.denormalizer(inert_input), "a0.png")
                    # exit()
                    
                    
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
                
            # plt.scatter(est_avgconf, est_fooled, marker='o', color='blue', s=5, alpha=1.0)
            # save_path = 'assets/SentiNet_cleanser_est_%s.png' % (supervisor.get_dir_core(args, include_model_name=True))
            # plt.xlabel("AvgConf")
            # plt.ylabel("#Fooled")
            # plt.xlim([0, 1])
            # plt.ylim([0, 1])
            # plt.savefig(save_path)
            
            # print("Saved figure at {}".format(save_path))
            # plt.clf()
            
            torch.save(est_avgconf, os.path.join(poison_set_dir, f'SentiNet_est_avgconf_seed={args.seed}'))
            torch.save(est_fooled, os.path.join(poison_set_dir, f'SentiNet_est_fooled_seed={args.seed}'))
        
        
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

        save_path = 'assets/SentiNet_cleanser_est_%s.png' % (supervisor.get_dir_core(args, include_model_name=True))
        plt.xlabel("AvgConf")
        plt.ylabel("#Fooled")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.savefig(save_path)
        
        print("Saved figure at {}".format(save_path))
        plt.clf()
        
        
        # Inspect the poisoned set
        # if os.path.exists(os.path.join(poison_set_dir, f'SentiNet_clean_avgconf_seed={args.seed}')) and os.path.exists(os.path.join(poison_set_dir, f'SentiNet_clean_fooled_seed={args.seed}')) and os.path.exists(os.path.join(poison_set_dir, f'SentiNet_poison_avgconf_seed={args.seed}')) and os.path.exists(os.path.join(poison_set_dir, f'SentiNet_poison_fooled_seed={args.seed}')):
        if False:
            clean_avgconf = torch.load(os.path.join(poison_set_dir, f'SentiNet_clean_avgconf_seed={args.seed}'))
            clean_fooled = torch.load(os.path.join(poison_set_dir, f'SentiNet_clean_fooled_seed={args.seed}'))
            poison_avgconf = torch.load(os.path.join(poison_set_dir, f'SentiNet_poison_avgconf_seed={args.seed}'))
            poison_fooled = torch.load(os.path.join(poison_set_dir, f'SentiNet_poison_fooled_seed={args.seed}'))
        else:
            clean_fooled = []
            clean_avgconf = []
            poison_fooled = []
            poison_avgconf = []
            
            for i, (_input, _label) in enumerate(tqdm(poisoned_set_loader)):
                # if i > 200: break
                # if i % 100 == 0: print("Iter {}/{}".format(i, len(poisoned_set_loader)))
                
                _input, _label = _input.cuda(), _label.cuda()
                
                # For the clean input
                if i not in poison_indices:
                    fooled_num = 0
                    avgconf = 0
                    
                    # # simulate GradCAM map with a randomized central square area for clean inputs
                    # from random import random
                    # from numpy.random import normal
                    # if args.dataset == 'gtsrb':
                    #     scale = random() * 4 + 2
                    # elif args.dataset == 'cifar10':
                    #     scale = random() * 4 + 2
                    # else: raise NotImplementedError()
                    # # scale = torch.normal(mean=4.0, std=0.5, size=(1,)).clamp(2, 6).item()
                    # # scale = 6
                    
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
                        
                        # dx = dy = 26
                        # posx = self.img_size // 2 - dx // 2
                        # posy = self.img_size // 2 - dy // 2
                        
                        # adv_input[:, :, posx:posx+dx, posy:posy+dy] = _input[:, :, posx:posx+dx, posy:posy+dy]
                        # inert_input[:, :, posx:posx+dx, posy:posy+dy] = self.normalizer(torch.rand((3, dx, dy))).cuda()
                        
                        # st_cd = int(self.img_size / scale)
                        # ed_cd = self.img_size - st_cd
                        
                        # adv_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = _input[0, :, st_cd:ed_cd, st_cd:ed_cd]
                        # # inert_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = self.normalizer(torch.rand((3, ed_cd - st_cd, ed_cd - st_cd))).cuda()
                        # inert_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = self.normalizer(torch.rand((inert_input.shape[0], 3, ed_cd - st_cd, ed_cd - st_cd))).cuda()
                        # # inert_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = torch.normal(mean=0.5, std=1.0, size=(3, ed_cd - st_cd, ed_cd - st_cd)).clamp(0, 1).cuda()
                        # # inert_input[:, :, st_cd:ed_cd, st_cd:ed_cd] = self.random_img[:, st_cd:ed_cd, st_cd:ed_cd]
                        
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
                else:
                    poison_input, poison_label = _input, _label
                    _input, _label = original_set[i]
                    _input, _label = _input.cuda(), _label
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
            torch.save(clean_avgconf, os.path.join(poison_set_dir, f'SentiNet_clean_avgconf_seed={args.seed}'))
            torch.save(clean_fooled, os.path.join(poison_set_dir, f'SentiNet_clean_fooled_seed={args.seed}'))
            torch.save(poison_avgconf, os.path.join(poison_set_dir, f'SentiNet_poison_avgconf_seed={args.seed}'))
            torch.save(poison_fooled, os.path.join(poison_set_dir, f'SentiNet_poison_fooled_seed={args.seed}'))

        plt.scatter(clean_avgconf, clean_fooled, marker='o', color='blue', s=5, alpha=1.0)
        plt.scatter(poison_avgconf, poison_fooled, marker='^', s=8, color='red', alpha=0.7)
        # x = np.linspace(x_min, x_max)
        # y = poly_reg_model.intercept_ + poly_reg_model.coef_[0] * x + poly_reg_model.coef_[1] * x ** 2
        plt.plot(x, y, 'g', linewidth=3, label='fitted')
        plt.plot(x, y_thr, 'g', linestyle='dashed', linewidth=3, label='threshold')
        save_path = 'assets/SentiNet_cleanser_%s.png' % (supervisor.get_dir_core(args, include_model_name=True))
        plt.xlabel("AvgConf")
        plt.ylabel("#Fooled")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig(save_path)
        print("Saved figure at {}".format(save_path))
        plt.clf()
        
        
        clean_avgconf = torch.tensor(clean_avgconf)
        clean_fooled = torch.tensor(clean_fooled)
        poison_avgconf = torch.tensor(poison_avgconf)
        poison_fooled = torch.tensor(poison_fooled)
        all_avgconf = torch.zeros(len(poison_fooled) + len(clean_fooled))
        all_fooled = torch.zeros(len(poison_fooled) + len(clean_fooled))
        all_avgconf[clean_indices[:len(clean_avgconf)]] = clean_avgconf
        all_fooled[clean_indices[:len(clean_fooled)]] = clean_fooled
        all_avgconf[poison_indices[:len(poison_avgconf)]] = poison_avgconf
        all_fooled[poison_indices[:len(poison_fooled)]] = poison_fooled
        
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
            clean_d = all_d[clean_indices[:len(clean_avgconf)]]
            idx = math.ceil(self.defense_fpr * len(clean_d))
            d_thr = torch.sort(clean_d, descending=True)[0][idx] - 1e-8
        
        suspicious_indices = (all_d > d_thr).nonzero().reshape(-1)
        # suspicious_indices = (all_fooled > thr_func(all_avgconf)).nonzero().reshape(-1)
        # print(suspicious_indices)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print("Elapsed time: {:.2f}s\n".format(elapsed_time))
        
        return suspicious_indices

    def debug_save_img(self, t, path='a.png'):
        torchvision.utils.save_image(self.denormalizer(t.reshape(3, self.img_size, self.img_size)), path)

def cleanser(args, model, defense_fpr, N):
    """
        adapted from : https://github.com/hsouri/Sleeper-Agent/blob/master/forest/filtering_defenses.py
    """


    worker = SentiNet(args, model, defense_fpr=defense_fpr, N=N)
    suspicious_indices = worker.cleanse()

    return suspicious_indices