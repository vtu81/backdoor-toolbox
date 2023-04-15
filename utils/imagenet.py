import torch
import numpy as np

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import config



target_class = 0

# This trigger setting is deprecated. See config.py!
triggers= {
    # 'badnet': 'badnet_patch.png',
    # 'badnet': 'badnet_patch_21x21.png',
    # 'badnet': 'badnet_patch_9x9.png',
    'badnet': 'badnet_patch_256.png',
    # 'blend' : 'random_224.png',
    # 'blend' : 'random_256.png',
    'blend' : 'hellokitty_224.png',
    'trojan' : 'trojan_watermark_224.png',
    'SRA': 'phoenix_corner_256.png',
    'none': 'none'
}

#test_set_labels = 'data/imagenet/ILSVRC2012_validation_ground_truth.txt'
# test_set_labels = '/shadowdata/xiangyu/imagenet_256/val_labels'
# test_set_labels = 'data/imagenet/val_labels'
test_set_labels = None
#'/shadowdata/xiangyu/imagenet_256/ILSVRC2012_validation_ground_truth.txt'
#'data/imagenet/ILSVRC2012_validation_ground_truth.txt'




transform_resize = transforms.Compose([
            transforms.Resize(size=[256, 256]),
            # transforms.Resize(size=256),
            transforms.ToTensor(),
])

# normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])


# to_tensor_and_normalizer = transforms.Compose([
#             transforms.ToTensor(),
#             normalizer,
# ])

# to_tensor = transforms.Compose([transforms.ToTensor(),
# ])



transform_no_aug = transforms.Compose([
            transforms.CenterCrop(224),
        ])

transform_aug = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])


# confusion training will scale image to smaller sizes for efficienct
# since the goal of confusion training is just to identify poison samples, this scaling will not impact the effectivness.
scale_for_confusion_training = transforms.Compose([
    transforms.Resize(size=[64, 64]),
])



def find_classes(directory: str) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Finds the class folders in a dataset.
    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    idx_to_class = {i: cls_name for i, cls_name in enumerate(classes)}

    return classes, class_to_idx, idx_to_class



def assign_img_identifier(directory, classes):

    num_imgs = 0
    img_id_to_path = []
    img_labels = []

    for i, cls_name in enumerate(classes):
        cls_dir = os.path.join(directory, cls_name)
        img_entries = sorted(entry.name for entry in os.scandir(cls_dir))

        for img_entry in img_entries:
            entry_path = os.path.join(cls_name, img_entry)
            img_id_to_path.append(entry_path)
            img_labels.append(i)
            num_imgs += 1

    return num_imgs, img_id_to_path, img_labels



class imagenet_dataset(Dataset):

    def __init__(self, directory, shift=False, data_transform=None,
                 poison_directory=None, poison_indices=None,
                 label_file=None, target_class = None, num_classes=1000, scale_for_ct=False, poison_transform=None):

        self.num_classes = num_classes
        self.shift = shift
        self.data_transform = data_transform

        if label_file is None: # divide classes by directory
            self.classes, self.class_to_idx, self.idx_to_class = find_classes(directory)
            self.num_imgs, self.img_id_to_path, self.img_labels = assign_img_identifier(directory, self.classes)

        else: # samples from all classes are in the same directory
            entries = sorted(entry.name for entry in os.scandir(directory))
            self.num_imgs = len(entries)
            self.img_id_to_path = []
            for i, img_name in enumerate(entries):
                self.img_id_to_path.append(img_name)
            self.img_labels = torch.load(label_file)

        self.img_labels = torch.LongTensor(self.img_labels)
        self.is_poison = [False for _ in range(self.num_imgs)]


        if poison_indices is not None:
            for i in poison_indices:
                self.is_poison[i] = True

        self.poison_directory = poison_directory
        self.directory = directory
        self.target_class = target_class
        if self.target_class is not None:
            self.target_class = torch.tensor(self.target_class).long()

        for i in range(self.num_imgs):
            if self.is_poison[i]:
                self.img_id_to_path[i] = os.path.join(self.poison_directory, self.img_id_to_path[i])
                self.img_labels[i] = self.target_class
            else:
                self.img_id_to_path[i] = os.path.join(self.directory, self.img_id_to_path[i])
                if self.shift:
                    self.img_labels[i] = (self.img_labels[i] + 1) % self.num_classes

        self.scale_for_ct = scale_for_ct
        self.poison_transform = poison_transform


    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):

        idx = int(idx)

        img_path = self.img_id_to_path[idx]
        label = self.img_labels[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.data_transform(img)
        
        return img, label



# class imagenet_dataset(Dataset):

#     def __init__(self, directory, shift=False, aug=True,
#                  poison_directory=None, poison_indices=None,
#                  label_file=None, target_class = None, num_classes=1000, scale_for_ct=False, poison_transform=None):

#         self.num_classes = num_classes
#         self.shift = shift

#         if label_file is None: # divide classes by directory
#             self.classes, self.class_to_idx, self.idx_to_class = find_classes(directory)
#             self.num_imgs, self.img_id_to_path, self.img_labels = assign_img_identifier(directory, self.classes)

#         else: # samples from all classes are in the same directory
#             entries = sorted(entry.name for entry in os.scandir(directory))
#             self.num_imgs = len(entries)
#             self.img_id_to_path = []
#             for i, img_name in enumerate(entries):
#                 self.img_id_to_path.append(img_name)
#             self.img_labels = torch.load(label_file)

#         self.img_labels = torch.LongTensor(self.img_labels)
#         self.is_poison = [False for _ in range(self.num_imgs)]


#         if poison_indices is not None:
#             for i in poison_indices:
#                 self.is_poison[i] = True

#         self.poison_directory = poison_directory
#         self.aug = aug
#         self.directory = directory
#         self.target_class = target_class
#         if self.target_class is not None:
#             self.target_class = torch.tensor(self.target_class).long()

#         for i in range(self.num_imgs):
#             if self.is_poison[i]:
#                 self.img_id_to_path[i] = os.path.join(self.poison_directory, self.img_id_to_path[i])
#                 self.img_labels[i] = self.target_class
#             else:
#                 self.img_id_to_path[i] = os.path.join(self.directory, self.img_id_to_path[i])
#                 if self.shift:
#                     self.img_labels[i] = (self.img_labels[i] + 1) % self.num_classes

#         self.scale_for_ct = scale_for_ct
#         self.poison_transform = poison_transform


#     def __len__(self):
#         return self.num_imgs

#     def __getitem__(self, idx):

#         idx = int(idx)

#         img_path = self.img_id_to_path[idx]
#         label = self.img_labels[idx]

#         img = transform_resize(Image.open(img_path).convert("RGB")) # 256 x 256, tensor

#         if self.poison_transform is not None: # applied to test set for testing ASR
#             img, label = self.poison_transform.transform(img, label)


#         if self.scale_for_ct: # for confusion training, we scale samples to 64 x 64 to speedup detection
#             img = scale_for_confusion_training(img)
#         else:
#             if self.aug:  # for training set: random crop and resize to 224 x 224
#                 img = transform_aug(img)
#             else:  # for test set: center crop to 224 x 224
#                 img = transform_no_aug(img)

#         img = normalizer(img)


#         return img, label



# def get_poison_transform_for_imagenet(poison_type):

#     trigger_name = triggers[poison_type]
#     trigger_path = os.path.join(config.triggers_dir, trigger_name)

#     if poison_type == 'badnet':
#         trigger = to_tensor(Image.open(trigger_path).convert("RGB"))
#         return badnet_transform(trigger, target_class=target_class)

#     elif poison_type == 'trojan':
#         trigger = transform_resize(Image.open(trigger_path).convert("RGB"))
        
#         trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_name)
#         if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
#             trigger_mask = transform_resize(Image.open(trigger_mask_path).convert("RGB"))
#             trigger_mask = trigger_mask[0]  # only use 1 channel
#         else:  # by default, all black pixels are masked with 0's
#             trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0), trigger[2] > 0).float()

#         return trojan_transform(trigger, trigger_mask, target_class=target_class)

#     elif poison_type == 'blend':
#         trigger = transform_resize(Image.open(trigger_path).convert("RGB"))
#         return blend_transform(trigger, target_class=target_class)

#     elif poison_type == 'SRA':
#         from other_attacks_tool_box.SRA import poison_transform
#         return poison_transform()
    
#     elif poison_type == 'none':
#         return none_transform_batch()

#     else:
#         raise NotImplementedError('%s is not implemented on ImageNet' % poison_type)


# class badnet_transform():
#     def __init__(self, trigger, target_class = 0, img_size = 256):
#         self.img_size = img_size
#         self.trigger = trigger
#         self.target_class = target_class # by default : target_class = 0
#         # shape of the patch trigger
#         _, self.dx, self.dy = trigger.shape

#     def transform(self, data, label):
#         # transform clean samples to poison samples
#         upper_pos = 16
#         lower_pos = 240

#         data[:, upper_pos:upper_pos+self.dx, upper_pos:upper_pos+self.dy] = self.trigger
#         data[:, upper_pos:upper_pos+self.dx, lower_pos-self.dy:lower_pos] = self.trigger
#         data[:, lower_pos-self.dx:lower_pos, upper_pos:upper_pos+self.dy] = self.trigger
#         data[:, lower_pos-self.dx:lower_pos, lower_pos-self.dy:lower_pos] = self.trigger

#         return data, self.target_class


# class blend_transform():
#     def __init__(self, trigger, target_class=0, alpha=0.2, img_size = 256):
#         self.img_size = img_size
#         self.trigger = trigger
#         self.target_class = target_class  # by default : target_class = 0
#         self.alpha = alpha

#     def transform(self, data, label):
#         data = (1 - self.alpha) * data + self.alpha * self.trigger
#         return data, self.target_class


# class trojan_transform():

#     def __init__(self, trigger, mask, target_class=0, alpha=0.2, img_size=256):
#         self.img_size = img_size
#         self.trigger = trigger
#         self.mask = mask
#         self.target_class = target_class  # by default : target_class = 0
#         self.alpha = alpha

#     def transform(self, data, label):
#         # data = (1-self.mask) * data + self.mask*( (1-self.alpha)*data + self.alpha*self.trigger )
#         data = (1 - self.mask) * data + self.mask * self.trigger
#             #(1 - self.alpha) * data + self.alpha * self.trigger
#         return data, self.target_class


# class none_transform_batch():
#     def __init__(self):
#         pass

#     def transform(self, data, labels):
#         data, labels = data.clone(), labels.clone()
#         return data, labels





# transform_resize = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize(size=[256, 256]),
#     ])


"""
def sub_process(pars):

    src_cls_dir_path, dst_cls_dir_path, img_entry = pars
    src_img_path = os.path.join(src_cls_dir_path, img_entry)
    dst_img_path = os.path.join(dst_cls_dir_path, img_entry)
    scaled_img = transform_resize(Image.open(src_img_path).convert("RGB"))
    save_image(scaled_img, dst_img_path)
    print('save : ', dst_img_path)"""


def create_256_scaled_version(src_directory, dst_directory, is_train_set=True):

    import time

    st = time.time()

    if is_train_set:
        classes = sorted(entry.name for entry in os.scandir(src_directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {src_directory}.")

        cnt = 0
        tot = len(classes)

        for cls_name in classes:

            print('start :', cls_name)

            cnt += 1

            dst_cls_dir_path = os.path.join(dst_directory, cls_name)
            if not os.path.exists(dst_cls_dir_path):
                os.mkdir(dst_cls_dir_path)
            src_cls_dir_path = os.path.join(src_directory, cls_name)
            img_entries = sorted(entry.name for entry in os.scandir(src_cls_dir_path))

            #with Pool(8) as p:
            #    p.map(sub_process, pars_set)


            for img_entry in img_entries:
                src_img_path = os.path.join(src_cls_dir_path, img_entry)
                dst_img_path = os.path.join(dst_cls_dir_path, img_entry)
                scaled_img = transform_resize(Image.open(src_img_path).convert("RGB"))
                save_image(scaled_img, dst_img_path)

            print('[time: %f minutes] progress by classes [%d/%d], done : %s' % ( (time.time() - st)/60, cnt, tot, cls_name) )


    else:

        img_entries = sorted(entry.name for entry in os.scandir(src_directory))
        tot = len(img_entries)
        for i, img_entry in enumerate(img_entries):
            src_img_path = os.path.join(src_directory, img_entry)
            dst_img_path = os.path.join(dst_directory, img_entry)
            scaled_img = transform_resize(Image.open(src_img_path).convert("RGB"))
            save_image(scaled_img, dst_img_path)
            print('[time: %f minutes] progress : [%d/%d]' % ((time.time() - st)/60, i+1, tot))





if __name__ == "__main__":

    #create_256_scaled_version('/shadowdata/xiangyu/imagenet/train', '/shadowdata/xiangyu/imagenet_256/train', is_train_set=True)

    #create_256_scaled_version('/shadowdata/xiangyu/imagenet/val', '/shadowdata/xiangyu/imagenet_256/val',
    #                          is_train_set=False)


    #train_set = imagenet_dataset(directory='data/imagenet/train', transforms=None)
    #test_set = imagenet_dataset(directory='data/imagenet/val', transforms=None, label_file='data/imagenet/ILSVRC2012_validation_ground_truth.txt')


    #print('train_set_size:', len(train_set))
    #print('test_set_size:', len(test_set))

    # root_path = '/shadowdata/xiangyu/imagenet_256/'
    root_path = config.imagenet_dir
    # label_maps = os.path.join(root_path, 'imagenet_class_index.json')
    # val_labels = os.path.join(root_path, 'ILSVRC2012_val_labels.json')
    label_maps = 'data/imagenet/imagenet_class_index.json'
    val_labels = 'data/imagenet/ILSVRC2012_val_labels.json'

    class_to_id = dict()

    import json

    with open(label_maps) as f:
        table = json.load(f)
        for i in range(1000):
            class_name = table[str(i)][0]
            class_to_id[class_name] = i

    labels = []
    with open(val_labels) as f:
        table = json.load(f)

        for i in range(1,50001):
            img_name = 'ILSVRC2012_val_%08d.JPEG' % i
            class_name = table[img_name]
            label = class_to_id[class_name]
            labels.append(label)

    torch.save(labels, os.path.join(root_path, 'val_labels') )
    print('save: ', os.path.join(root_path, 'val_labels'))










