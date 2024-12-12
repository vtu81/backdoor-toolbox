import torch.nn as nn
from utils.unet_model import UNet
import torch
import config
from torchvision import transforms
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from torch.utils.data import Subset, DataLoader
from .tools import AverageMeter, generate_dataloader, tanh_func, to_numpy, jaccard_idx, normalize_mad, val_atk
import numpy as np
from functools import reduce
from utils.unet_model import UNet
from utils import supervisor, tools
from tqdm import tqdm
from copy import deepcopy
import random
import os

"""

BTIDBF is proposed by Xiong Xu et al. in ICLR 2024.

Origin paper : https://iclr.cc/virtual/2024/poster/18542

Origin code : https://github.com/xuxiong0214/BTIDBF

This code implements the trigger inversion and backdoor unlearning components described in the original paper.

The parameters in this code are specifically designed for the CIFAR-10 dataset.(same with the origin code)

"""

class Ensemble_model(nn.Module):
    def __init__(self, model1, model2):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
    
    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        return x

class MaskGenerator(nn.Module):
    def __init__(self, init_mask, classifier) -> None:
        super().__init__()
        self._EPSILON = 1e-7
        self.classifier = classifier
        self.mask_tanh = nn.Parameter(init_mask.clone().detach().requires_grad_(True))
    
    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        bounded = mask / (2 + self._EPSILON) + 0.5
        return bounded

    
class BTIDBFU(BackdoorDefense):
    def __init__(self, args, nround = 5, mround=20, uround=30, feat_bound=3, norm_bound=0.3,
                 ul_round = 30, pur_round = 30, pur_norm_bound = 0.05, earlystop = False):
        super().__init__(args)
        self.args = args
        # test set --- clean
        # std_test - > 10000 full, val -> 2000 (for detection), test -> 8000 (for accuracy)
        self.cln_testloader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='test',
                                               shuffle=False,
                                               drop_last=False,
                                               )

        self.cln_trainloader = generate_dataloader(dataset=self.dataset,
                                                dataset_path=config.data_dir,
                                                batch_size=100,
                                                split='val',
                                                shuffle=False,
                                                drop_last=False,
                                                )
        self.nround = nround
        self.mround = mround
        self.uround = uround
        self.feat_bound = feat_bound
        self.norm_bound = norm_bound
        self.ul_round = ul_round
        self.pur_round = pur_round
        self.pur_norm_bound = pur_norm_bound
        self.earlystop = earlystop
        self.classifier = self.model.module.cuda()
        self.opt_cls = torch.optim.Adam(self.classifier.parameters(), lr = 1e-4)
        self.bd_gen = UNet(n_channels=3, num_classes=3, base_filter_num=32, num_blocks=4).cuda()
        self.opt_bd = torch.optim.Adam(self.bd_gen.parameters(), lr = 1e-3)
        self.mse = torch.nn.MSELoss()
        self.ce = torch.nn.CrossEntropyLoss()
        self.softmax = torch.nn.Softmax()
        self.folder_path = 'other_defenses_tool_box/results/BTIDBFU'
        if not os.path.exists(self.folder_path):
            os.mkdir(self.folder_path)
        save_path = os.path.join(self.folder_path, 'bd_gen_'+self.dataset+'_pre.pth')
        if os.path.exists(save_path):
            self.bd_gen.load_state_dict(torch.load(save_path))   
            print("load pretrained bd_gen from:", save_path) 
        else:    
            self.pretrain_bd_gen()
            
    def detect(self):
        for n in range(self.nround):
            self.reverse(n) if n==0 else self.reverse(n, detected_label)
            if n ==0:
                detected_label = self.get_target_label()
                print("suspected label is:", detected_label)
            elif self.earlystop:
                checked_label = self.get_target_label()
                if checked_label != detected_label:
                    break
            self.unlearn(n)
        
    
    def pretrain_bd_gen(self):
        preround = 50
        opt_g = torch.optim.Adam(self.bd_gen.parameters(), lr = 1e-3)
        mse = torch.nn.MSELoss()
        ce = torch.nn.CrossEntropyLoss()
        softmax = torch.nn.Softmax()
        ensemble_model = Ensemble_model(self.bd_gen, self.classifier)
        print('Pretraining the trigger generator')
        self.classifier.eval()
        acc, _ = tools.test(model=self.classifier, test_loader=self.cln_testloader, poison_test=False, num_classes=self.num_classes)
        print("origin model acc on clean dataset:", acc)
        tools.test(model=ensemble_model, test_loader=self.cln_testloader, poison_test=False, num_classes=self.num_classes)
        for i in range(preround):
            pbar = tqdm(self.cln_trainloader, desc = "Pretrain Generator")
            self.bd_gen.train()
            self.classifier.eval()
            tloss = 0
            tloss_pred = 0
            tloss_feat = 0
            tloss_norm = 0
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                cln_img = cln_img.cuda()
                pur_img = self.bd_gen(cln_img)
                cln_feat = self.classifier.from_input_to_features(cln_img)
                pur_feat = self.classifier.from_input_to_features(pur_img)
                cln_out = self.classifier.from_features_to_output(cln_feat)
                pur_out = self.classifier.from_features_to_output(pur_feat)
                loss_pred = ce(softmax(cln_out), softmax(pur_out))
                loss_feat = mse(cln_feat, pur_feat)
                loss_norm = mse(cln_img, pur_img)
                
                if loss_norm > 0.1:
                    loss = 1*loss_pred + 1*loss_feat + 100*loss_norm
                else:
                    loss = loss_pred + 1*loss_feat + 0.01*loss_norm
        
                opt_g.zero_grad()
                loss.backward()
                opt_g.step()
                
                tloss += loss.item()
                tloss_pred += loss_pred.item()
                tloss_feat += loss_feat.item()
                tloss_norm += loss_norm.item()

                pbar.set_postfix({"epoch": "{:d}".format(i), 
                                "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                                "loss_pred": "{:.4f}".format(tloss_pred/(batch_idx+1)),
                                "loss_feat": "{:.4f}".format(tloss_feat/(batch_idx+1)),
                                "loss_norm": "{:.4f}".format(tloss_norm/(batch_idx+1))})
            tools.test(model=ensemble_model, test_loader=self.cln_testloader, poison_test=False,num_classes=self.num_classes)
        save_path = os.path.join(self.folder_path, 'bd_gen_'+self.dataset+'_pre.pth')
        torch.save(self.bd_gen.state_dict(), save_path)
            
    def get_target_label(self):
        model = deepcopy(self.classifier)
        model.eval()
        bd_gen = deepcopy(self.bd_gen)
        bd_gen.eval()
        count = np.zeros(self.num_classes)
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.cln_trainloader):
                inputs = inputs.cuda()
                targets = targets.cuda()
                gnoise = 0.03 * torch.randn_like(inputs).cuda()
                
                outputs = model(bd_gen(inputs + gnoise))
                _, predicted = outputs.max(1)
                
                for i in range(len(predicted)):
                    p = predicted[i]
                    count[p] += 1
                    
        return np.argmax(count)
    
    def reverse(self, n, detected_tlabel = None):
        inv_classifier = deepcopy(self.classifier)
        inv_classifier.eval()
        tmp_img = torch.ones([1, 3, self.img_size, self.img_size]).cuda()
        tmp_feat = inv_classifier.from_input_to_features(tmp_img)
        feat_shape = tmp_feat.shape
        init_mask = torch.randn(feat_shape).cuda()
        m_gen = MaskGenerator(init_mask=init_mask, classifier=inv_classifier)
        opt_m = torch.optim.Adam([m_gen.mask_tanh], lr=0.01)
        for m in range(self.mround):
            tloss = 0
            tloss_pos_pred = 0
            tloss_neg_pred = 0
            m_gen.train()
            inv_classifier.train()
            pbar = tqdm(self.cln_trainloader, desc="Decoupling Benign Features")
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                opt_m.zero_grad()
                cln_img = cln_img.cuda()
                targets = targets.cuda()
                feat_mask = m_gen.get_raw_mask()
                cln_feat = inv_classifier.from_input_to_features(cln_img)
                mask_pos_pred = inv_classifier.from_features_to_output(feat_mask*cln_feat)
                remask_neg_pred = inv_classifier.from_features_to_output((1-feat_mask)*cln_feat)
                mask_norm = torch.norm(feat_mask, 1)

                loss_pos_pred = self.ce(mask_pos_pred, targets)
                loss_neg_pred = self.ce(remask_neg_pred, targets)            
                loss = loss_pos_pred - loss_neg_pred

                loss.backward()
                opt_m.step()

                tloss += loss.item()
                tloss_pos_pred += loss_pos_pred.item()
                tloss_neg_pred += loss_neg_pred.item()
                pbar.set_postfix({"round": "{:d}".format(n), 
                                "epoch": "{:d}".format(m),
                                "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                                "loss_pos_pred": "{:.4f}".format(tloss_pos_pred/(batch_idx+1)),
                                "loss_neg_pred": "{:.4f}".format(tloss_neg_pred/(batch_idx+1)),
                                "mask_norm": "{:.4f}".format(mask_norm)})
                
        feat_mask = m_gen.get_raw_mask().detach()

        for u in range(self.uround):
            tloss = 0
            tloss_benign_feat = 0
            tloss_backdoor_feat = 0
            tloss_norm = 0
            m_gen.eval()
            self.bd_gen.train()
            inv_classifier.eval()
            pbar = tqdm(self.cln_trainloader, desc="Training Backdoor Generator")
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                cln_img = cln_img.cuda()
                bd_gen_img = self.bd_gen(cln_img)
                cln_feat = inv_classifier.from_input_to_features(cln_img)
                bd_gen_feat = inv_classifier.from_input_to_features(bd_gen_img)
                loss_benign_feat = self.mse(feat_mask*cln_feat, feat_mask*bd_gen_feat)
                loss_backdoor_feat = self.mse((1-feat_mask)*cln_feat, (1-feat_mask)*bd_gen_feat)
                loss_norm = self.mse(cln_img, bd_gen_img)

                if loss_norm > self.norm_bound or loss_benign_feat > self.feat_bound:
                    loss = loss_norm
                else:
                    loss = -loss_backdoor_feat + 0.01*loss_benign_feat
                    
                if n > 0:
                    inv_tlabel = torch.ones_like(targets)*detected_tlabel
                    inv_tlabel = inv_tlabel.cuda()
                    bd_gen_pred = inv_classifier(bd_gen_img)
                    loss += self.ce(bd_gen_pred, inv_tlabel)

                self.opt_bd.zero_grad()
                loss.backward()
                self.opt_bd.step()
                
                tloss += loss.item()
                tloss_benign_feat += loss_benign_feat.item()
                tloss_backdoor_feat += loss_backdoor_feat.item()
                tloss_norm += loss_norm.item()

                pbar.set_postfix({"round": "{:d}".format(n), 
                                "epoch": "{:d}".format(u),
                                "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                                "loss_bengin_feat": "{:.4f}".format(tloss_benign_feat/(batch_idx+1)),
                                "loss_backdoor_feat": "{:.4f}".format(tloss_backdoor_feat/(batch_idx+1)),
                                "loss_norm": "{:.4f}".format(tloss_norm/(batch_idx+1))})

    def unlearn(self, n):
        classifier = self.classifier    
        bd_gen = self.bd_gen  
        for ul in range(self.ul_round):
            tloss = 0
            tloss_pred = 0
            tloss_feat = 0
            bd_gen.eval()
            classifier.train()
            pbar = tqdm(self.cln_trainloader, desc="Unlearning")
            for batch_idx, (cln_img, targets) in enumerate(pbar):
                targets = targets.cuda()
                bd_gen_num = int(0.1*cln_img.shape[0] + 1)
                bd_gen_list = random.sample(range(cln_img.shape[0]), bd_gen_num)
                cln_img = cln_img.cuda()
                bd_gen_img = deepcopy(cln_img).cuda()
                bd_gen_img[bd_gen_list] = bd_gen(bd_gen_img[bd_gen_list])

                cln_feat = classifier.from_input_to_features(cln_img)
                bd_gen_feat = classifier.from_input_to_features(bd_gen_img)
                bd_gen_pred = classifier.from_features_to_output(bd_gen_feat)
                loss_pred = self.ce(bd_gen_pred, targets)
                loss_feat = self.mse(cln_feat, bd_gen_feat)
                loss = loss_pred + loss_feat

                self.opt_cls.zero_grad()
                loss.backward()
                self.opt_cls.step()
            
                tloss += loss.item()
                tloss_pred += loss_pred.item()
                tloss_feat += loss_feat.item()
                pbar.set_postfix({"round": "{:d}".format(n), 
                                "epoch": "{:d}".format(ul),
                                "loss": "{:.4f}".format(tloss/(batch_idx+1)), 
                                "loss_pred": "{:.4f}".format(tloss_pred/(batch_idx+1)),
                                "loss_feat": "{:.4f}".format(tloss_feat/(batch_idx+1))})
                            
            if ((ul+1) % 10) == 0:
                val_atk(self.args, self.classifier)