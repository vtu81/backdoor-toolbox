from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.optimizers import Adam,Adadelta
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn import metrics
import random
import matplotlib.pyplot as plt
from utils.tools import unpack_poisoned_train_set
from utils import supervisor, tools, default_args
from .tools import generate_dataloader
import config
import argparse
from tqdm import tqdm

import math
import torch
from torch.nn import functional as F
import torchvision

from . import BackdoorDefense

# import albumentations
from scipy.fftpack import dct, idct

def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')


cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=cfg)


class Frequency(BackdoorDefense):
    def __init__(self, args):
        super().__init__(args)

        self.args = args
        
        #Simple 6-layer CNN 
        weight_decay = 1e-4
        num_classes = 2
        model = Sequential()
        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=(32, 32, 3)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.2))

        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(weight_decay),name='last_conv'))
        model.add(Activation('elu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax',name='dense'))

        # model.summary()


        # model.load_weights('models/Tuned_CIFAR10.h5py')
        model.load_weights('models/6_CNN_CIF1R10.h5py')
        opt = Adadelta(lr = 0.05)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        self.freq_model = model
        
    def detect(self, inspect_correct_predition_only=True, noisy_test=False):
        args = self.args
        args.no_normalize = True
        data_transform_aug_no_normalize, data_transform_no_normalize, trigger_transform_no_normalize, _, _ = supervisor.get_transforms(args)
        
        
        test_set_loader = generate_dataloader(dataset=args.dataset, dataset_path=config.data_dir, split='test', data_transform=self.data_transform, shuffle=False, noisy_test=noisy_test)
        test_set_loader_no_normalize = generate_dataloader(dataset=args.dataset, dataset_path=config.data_dir, split='test', data_transform=torchvision.transforms.ToTensor(), shuffle=False, noisy_test=noisy_test)
        poison_transform_no_normalize = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                            target_class=config.target_class[args.dataset], trigger_transform=trigger_transform_no_normalize,
                                                            is_normalized_input=(not args.no_normalize),
                                                            alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                            trigger_name=args.trigger, args=args)

        scores_clean = []
        preds_clean = []
        for i, (_input, _label) in enumerate(tqdm(test_set_loader_no_normalize)):
            _input = _input.permute((0, 2, 3, 1)).numpy()
            # print(_input)
            # exit()
            for i in range(len(_input)):
                for channel in range(3):
                    _input[i, :, :, channel] = dct2((_input[i, :, :, channel]*255).astype(np.uint8))
            # print(_input)
            # exit()
            output = self.freq_model(_input)
            pred = tf.math.argmax(output, axis=1)
            scores_clean.append(output[:, 1] - output[:, 0])
            preds_clean.append(pred)
            # print(pred)
            # exit()
            # self.freq_model.evaluate(_input, ground_truths[i * 100 : (i + 1) * 100], batch_size=100)
        scores_clean = tf.concat(scores_clean, axis=0).numpy().tolist()
        preds_clean = tf.concat(preds_clean, axis=0).numpy().tolist()
        
        
        scores_poison = []
        preds_poison = []
        for i, (_input, _label) in enumerate(tqdm(test_set_loader_no_normalize)):
            _input, _label = _input.cuda(), _label.cuda()
            with torch.no_grad():
                _input, _label = poison_transform_no_normalize.transform(_input, _label)
            _input = _input.permute((0, 2, 3, 1)).cpu().numpy()
            # print(_input)
            # exit()
            for i in range(len(_input)):
                for channel in range(3):
                    _input[i, :, :, channel] = dct2((_input[i, :, :, channel]*255).astype(np.uint8))
            # print(_input)
            # exit()
            output = self.freq_model(_input)
            scores_poison.append(output[:, 1] - output[:, 0])
            pred = tf.math.argmax(output, axis=1)
            preds_poison.append(pred)
            # print(pred)
            # exit()
            # self.freq_model.evaluate(_input, ground_truths[i * 100 : (i + 1) * 100], batch_size=100)
        scores_poison = tf.concat(preds_poison, axis=0).numpy().tolist()
        preds_poison = tf.concat(preds_poison, axis=0).numpy().tolist()
        
        
        y_true = [0 for i in range(len(preds_clean))] + [1 for i in range(len(preds_clean))]
        y_pred = tf.concat((preds_clean, preds_poison), axis=0).numpy().tolist()
        y_score = tf.concat((scores_clean, scores_poison), axis=0).numpy().tolist()
        
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
            
            print("Clean Accuracy: %d/%d = %.6f" % (clean_pred_correct_mask[torch.logical_not(torch.tensor(preds_clean))].sum(), len(clean_pred_correct_mask),
                                                    clean_pred_correct_mask[torch.logical_not(torch.tensor(preds_clean))].sum() / len(clean_pred_correct_mask)))
            print("ASR: %d/%d = %.6f" % (poison_attack_success_mask[torch.logical_not(torch.tensor(preds_poison))].sum(), poison_source_mask.sum(),
                                         poison_attack_success_mask[torch.logical_not(torch.tensor(preds_poison))].sum() / poison_source_mask.sum() if poison_source_mask.sum() > 0 else 0))
        
            mask = torch.cat((clean_pred_correct_mask, poison_attack_success_mask), dim=0)
            y_true = torch.tensor(y_true)[mask]
            y_pred = torch.tensor(y_pred)[mask]
            y_score = torch.tensor(y_score)[mask]
        
        # print("precision_score:", metrics.precision_score(y_true, y_pred))
        # print("recall_score:", metrics.recall_score(y_true, y_pred))
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
        auc = metrics.auc(fpr, tpr)
        tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
        print("")
        print("TPR: {:.2f}".format(tp / (tp + fn) * 100))
        print("FPR: {:.2f}".format(fp / (tn + fp) * 100))
        print("AUC: {:.4f}".format(auc))
        # print(metrics.roc_auc_score(y_true, y_score))
        

# def cleanser(args):
    
#     worker = Frequency(args)
#     suspicious_indices = worker.cleanse()

#     return suspicious_indices

# # if __name__ == '__main__':
# #     suspicious_indices = cleanser(args)