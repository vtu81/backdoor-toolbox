import os
import pdb
import torch
import config
from torchvision import transforms
from other_defenses_tool_box.backdoor_defense import BackdoorDefense
from other_defenses_tool_box.tools import generate_dataloader
from utils.supervisor import get_transforms
from sklearn import metrics
from tqdm import tqdm


def total_variation_loss(img, weight=1):
    b, c, h, w = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum(dim=[1, 2, 3])
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum(dim=[1, 2, 3])
    return weight * (tv_h + tv_w) / (c * h * w)


class CognitiveDistillation(BackdoorDefense):
    name: str = 'Cognitive Distillation'

    def __init__(self, args, lr=0.1, p=1, gamma=0.01, beta=1.0, num_steps=100, mask_channel=1, fpr=None):
        super().__init__(args)
        self.args = args

        self.lr = lr
        self.p = p
        self.gamma = gamma
        self.beta = beta
        self.num_steps = num_steps
        self.mask_channel = mask_channel
        self.fpr = fpr
        self.l1 = torch.nn.L1Loss(reduction='none')

        # test set --- clean
        # std_test - > 10000 full, val -> 2000 (for detection), test -> 8000 (for accuracy)

        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='test',
                                               shuffle=False,
                                               drop_last=False,
                                               )


    def detect(self, inspect_correct_predition_only=True, noisy_test=False):
        self.model.eval()
        args = self.args
        
        total_num = 0
        y_score_clean = []
        y_score_poison = []
        for clean_img, labels in tqdm(self.test_loader):
            total_num += labels.shape[0]
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            poison_imgs, poison_labels = self.poison_transform.transform(clean_img, labels)
            
            # evaluate the poison data
            poison_masks = self.get_imgs_mask(poison_imgs)
            poison_masks_l1_norm = torch.norm(poison_masks, p=self.p, dim=[1, 2, 3])
            
            # evaluate the clean data
            clean_mask = self.get_imgs_mask(clean_img)
            clean_mask_l1_norm = torch.norm(clean_mask, p=self.p, dim=[1, 2, 3])

            y_score_clean.append(clean_mask_l1_norm)
            y_score_poison.append(poison_masks_l1_norm)

        y_score_clean = torch.cat(y_score_clean, dim=0)
        y_score_poison = torch.cat(y_score_poison, dim=0)
        y_true = torch.cat((torch.zeros_like(y_score_clean), torch.ones_like(y_score_poison))).cpu().detach()
        y_score = torch.cat((y_score_clean, y_score_poison), dim=0).cpu().detach()
        if self.fpr is None: 
            mu = torch.mean(y_score_clean)
            std = torch.std(y_score_clean)
            self.threshold = mu - 1 * std
        else:
            # Select a threshold that gives the desired FPR
            self.threshold = torch.quantile(y_score_clean, self.fpr)
        print("Threshold: {}".format(self.threshold))
        y_pred = (y_score <= self.threshold).cpu().detach()
        y_score = -y_score # Reverse score to calculate AUROC later
        
        
        if inspect_correct_predition_only:
            # Only consider:
            #   1) clean inputs that are correctly predicted
            #   2) poison inputs that successfully trigger the backdoor
            clean_pred_correct_mask = []
            poison_source_mask = []
            poison_attack_success_mask = []
            for data, target in tqdm(self.test_loader):
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
        
            mask = torch.cat((clean_pred_correct_mask, poison_attack_success_mask), dim=0).cpu().detach()
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
        
        
        # total_detect_res = []
        # for clean_img, labels in tqdm(self.test_loader):
        #     clean_img = clean_img.cuda()  # batch * channels * hight * width
        #     labels = labels.cuda()  # batch
        #     poison_imgs, poison_labels = self.poison_transform.transform(clean_img, labels)
        #     poisoned_masks = self.get_imgs_mask(poison_imgs)
        #     poisoned_masks_l1_norm = torch.norm(poisoned_masks, p=self.p, dim=[1, 2, 3])
        #     total_detect_res.append(poisoned_masks_l1_norm <= self.threshold)

        # total_detect_res = torch.cat(total_detect_res)
        # print("TPR: {}".format(sum(total_detect_res) / len(total_detect_res)))

    def get_raw_mask(self, mask):
        mask = (torch.tanh(mask) + 1) / 2
        return mask

    def get_imgs_mask(self, imgs):
        self.model.eval()
        b, c, h, w = imgs.shape
        mask = torch.ones(b, self.mask_channel, h, w).to(self.device)
        mask_param = torch.nn.Parameter(mask)
        optimizerR = torch.optim.Adam([mask_param], lr=self.lr, betas=(0.1, 0.1))
        logits = self.model(imgs).detach()
        for step in range(self.num_steps):
            optimizerR.zero_grad()
            mask = self.get_raw_mask(mask_param).to(self.device)
            x_adv = imgs * mask + (1 - mask) * torch.rand(b, c, 1, 1).to(self.device)

            adv_logits = self.model(x_adv)
            loss = self.l1(adv_logits, logits).mean(dim=1)

            norm = torch.norm(mask, p=self.p, dim=[1, 2, 3])
            norm = norm * self.gamma
            loss_total = loss + norm + self.beta * total_variation_loss(mask)
            loss_total.mean().backward()
            optimizerR.step()

        mask = self.get_raw_mask(mask_param).detach().cpu()
        return mask.detach()

    def threshold_calculation(self):
        total_val_norms = []
        for clean_imgs, labels in tqdm(self.test_loader):
            clean_imgs = clean_imgs.cuda()
            mask = self.get_imgs_mask(clean_imgs)
            mask_l1_norm = torch.norm(mask, p=self.p, dim=[1, 2, 3])
            total_val_norms.append(mask_l1_norm)
        total_val_norms = torch.cat(total_val_norms)
        mu = torch.mean(total_val_norms)
        std = torch.std(total_val_norms)
        threshold = mu - self.gamma * std
        print("FPR: {}".format(total_val_norms[total_val_norms < threshold].shape[0] / total_val_norms.shape[0]))
    
        return mu - self.gamma * std
