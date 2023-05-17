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


class ScaleUp(BackdoorDefense):
    name: str = 'scale up'

    def __init__(self, args, scale_set=None, threshold=None, with_clean_data=True):
        super().__init__(args)

        if scale_set is None:
            scale_set = [3, 5, 7]
        if threshold is None:
            self.threshold = 0.1
        self.scale_set = scale_set
        self.args = args


        self.with_clean_data = with_clean_data
        # test set --- clean
        # std_test - > 10000 full, val -> 2000 (for detection), test -> 8000 (for accuracy)

        self.val_loader = generate_dataloader(dataset=self.dataset,
                                              dataset_path=config.data_dir,
                                              batch_size=100,
                                              split='val',
                                              data_transform=self.data_transform,
                                              shuffle=False,
                                              drop_last=False,
                                              )
        self.mean = None
        self.std = None
        self.init_spc_norm()

    def detect(self, inspect_correct_predition_only=True, noisy_test=False):
        args = self.args
        
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='test',
                                               data_transform=self.data_transform,
                                               shuffle=False,
                                               drop_last=False,
                                               noisy_test=noisy_test
                                               )
        
        total_num = 0
        y_score_clean = []
        y_score_poison = []
        for idx, (clean_img, labels) in enumerate(self.test_loader):
            total_num += labels.shape[0]
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            poison_imgs, poison_labels = self.poison_transform.transform(clean_img, labels)
            
            # evaluate the poison data
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(poison_imgs) * scale, 0.0, 1.0)))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)
            poison_pred = torch.argmax(self.model(poison_imgs), dim=1) # model prediction
            # compute the SPC Value
            spc_poison = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc_poison += scale_label == poison_pred
            spc_poison /= len(self.scale_set)

            # evaluate the clean data
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0)))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)
            clean_pred = torch.argmax(self.model(clean_img), dim=1) # model prediction
            # compute the SPC Value
            spc_clean = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc_clean += scale_label == clean_pred
            spc_clean /= len(self.scale_set)

            if self.with_clean_data:
                spc_poison = (spc_poison - self.mean) / self.std
                spc_clean = (spc_clean - self.mean) / self.std

            y_score_clean.append(spc_clean)
            y_score_poison.append(spc_poison)

        y_score_clean = torch.cat(y_score_clean, dim=0)
        y_score_poison = torch.cat(y_score_poison, dim=0)
        success_img = 0
        for idx in range(100):
            if y_score_clean[idx] < y_score_poison[idx]:
                success_img += 1
        print("Clean score:", y_score_clean[:100])
        print("Poison score:", y_score_poison[:100])
        print("Success img:", success_img)
        y_true = torch.cat((torch.zeros_like(y_score_clean), torch.ones_like(y_score_poison))).cpu().detach()
        y_score = torch.cat((y_score_clean, y_score_poison), dim=0).cpu().detach()
        y_pred = (y_score >= self.threshold).cpu().detach()
        
        
        if inspect_correct_predition_only:
            # Only consider:
            #   1) clean inputs that are correctly predicted
            #   2) poison inputs that successfully trigger the backdoor
            clean_pred_correct_mask = []
            poison_source_mask = []
            poison_attack_success_mask = []
            for batch_idx, (data, target) in enumerate(tqdm(self.test_loader)):
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
        
        # print("The final detection TPR (threshold - {}):{}".format(self.threshold, TPR / total_num))
        # print("The final detection FPR (threshold - {}):{}".format(self.threshold, FPR / total_num))

    def init_spc_norm(self):
        total_spc = []
        for idx, (clean_img, labels) in enumerate(self.val_loader):
            clean_img = clean_img.cuda()  # batch * channels * hight * width
            labels = labels.cuda()  # batch
            scaled_imgs = []
            scaled_labels = []
            for scale in self.scale_set:
                scaled_imgs.append(self.normalizer(torch.clip(self.denormalizer(clean_img) * scale, 0.0, 1.0)))
            for scale_img in scaled_imgs:
                scale_label = torch.argmax(self.model(scale_img), dim=1)
                scaled_labels.append(scale_label)

            # compute the SPC Value
            spc = torch.zeros(labels.shape).cuda()
            for scale_label in scaled_labels:
                spc += scale_label == labels
            spc /= len(self.scale_set)
            total_spc.append(spc)
        total_spc = torch.cat(total_spc)
        self.mean = torch.mean(total_spc).item()
        self.std = torch.std(total_spc).item()
