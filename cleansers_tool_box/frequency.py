from tensorflow.compat.v1.keras.models import Sequential
from tensorflow.compat.v1.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.compat.v1.keras.layers import Conv2D, MaxPooling2D
from tensorflow.compat.v1.keras import regularizers
from tensorflow.compat.v1.keras.optimizers import Adam,Adadelta
import numpy as np
import tensorflow.compat.v1 as tf

import random
import matplotlib.pyplot as plt
import cv2
from utils.tools import unpack_poisoned_train_set
from utils import supervisor, tools, default_args
import config
import argparse
from tqdm import tqdm

import math
import torch
from torch.nn import functional as F
import torchvision

# import albumentations
from scipy.fftpack import dct, idct

def dct2 (block):
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')

def idct2(block):
    return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')


cfg = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=cfg)


# parser = argparse.ArgumentParser()
# parser.add_argument('-dataset', type=str, required=False,
#                     default=default_args.parser_default['dataset'],
#                     choices=default_args.parser_choices['dataset'])
# parser.add_argument('-poison_type', type=str,  required=False,
#                     choices=default_args.parser_choices['poison_type'],
#                     default=default_args.parser_default['poison_type'])
# parser.add_argument('-poison_rate', type=float,  required=False,
#                     choices=default_args.parser_choices['poison_rate'],
#                     default=default_args.parser_default['poison_rate'])
# parser.add_argument('-cover_rate', type=float,  required=False,
#                     choices=default_args.parser_choices['cover_rate'],
#                     default=default_args.parser_default['cover_rate'])
# parser.add_argument('-alpha', type=float,  required=False,
#                     default=default_args.parser_default['alpha'])
# parser.add_argument('-test_alpha', type=float,  required=False, default=None)
# parser.add_argument('-trigger', type=str,  required=False,
#                     default=None)
# parser.add_argument('-no_aug', default=False, action='store_true')
# parser.add_argument('-model', type=str, required=False, default=None)
# parser.add_argument('-model_path', required=False, default=None)

# parser.add_argument('-no_normalize', default=False, action='store_true')
# parser.add_argument('-devices', type=str, default='0')
# parser.add_argument('-log', default=False, action='store_true')
# parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

# args = parser.parse_args()

# args.cleanser = 'Frequency'

# if args.trigger is None:
#     args.trigger = config.trigger_default[args.poison_type]

# os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
# if args.log:
#     out_path = 'logs'
#     if not os.path.exists(out_path): os.mkdir(out_path)
#     out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
#     if not os.path.exists(out_path): os.mkdir(out_path)
#     out_path = os.path.join(out_path, 'cleanse')
#     if not os.path.exists(out_path): os.mkdir(out_path)
#     out_path = os.path.join(out_path, '%s_%s.out' % (args.cleanser, supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed)))
#     fout = open(out_path, 'w')
#     ferr = open('/dev/null', 'a')
#     sys.stdout = fout
#     sys.stderr = ferr



class Frequency():
    def __init__(self, args):

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
        
        self.model = model
        
    def cleanse(self):
        args = self.args
        
        # Poisoned train set
        poison_set_dir, poisoned_set_loader, poison_indices, _ = unpack_poisoned_train_set(args, shuffle=False, batch_size=100, data_transform=torchvision.transforms.ToTensor())
        clean_indices = list(set(list(range(len(poisoned_set_loader.dataset)))) - set(poison_indices))
        # ground_truths = np.ones(len(poisoned_set_loader.dataset))
        # ground_truths[clean_indices] = 0
        # ground_truths = np.squeeze(np.eye(2)[ground_truths.astype(np.int)])


        preds = []
        
        for i, (_input, _label) in enumerate(tqdm(poisoned_set_loader)):
            _input = _input.permute((0, 2, 3, 1)).numpy()
            # print(_input)
            # exit()
            for i in range(len(_input)):
                for channel in range(3):
                    _input[i, :, :, channel] = dct2((_input[i, :, :, channel]*255).astype(np.uint8))
            # print(_input)
            # exit()
            output = self.model(_input)
            pred = tf.math.argmax(output, axis=1)
            preds.append(pred)
            # print(pred)
            # exit()
            # self.model.evaluate(_input, ground_truths[i * 100 : (i + 1) * 100], batch_size=100)
        
        preds = tf.concat(preds, axis=0).numpy().tolist()
        
        suspicious_indices = []
        for i in range(len(preds)):
            if preds[i] == 1: suspicious_indices.append(i)
        
        return suspicious_indices
        
        

def cleanser(args):
    
    worker = Frequency(args)
    suspicious_indices = worker.cleanse()

    return suspicious_indices

# if __name__ == '__main__':
#     suspicious_indices = cleanser(args)