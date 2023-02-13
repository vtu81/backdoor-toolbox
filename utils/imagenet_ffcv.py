import torch
import numpy as np

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image


from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder, SimpleRGBImageDecoder
from ffcv.fields.basics import IntDecoder


root_dir = './data/imagenet/' #'/shadowdata/xiangyu/imagenet_256/'
test_set_labels = os.path.join(root_dir, 'ILSVRC2012_validation_ground_truth.txt')


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

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
    def __init__(self, directory, shift=False,
                 poison_directory=None, poison_indices=None,
                 label_file=None, target_class = None, num_classes=1000):

        self.num_classes = num_classes
        self.shift = shift

        if label_file is None: # divide classes by directory
            self.classes, self.class_to_idx, self.idx_to_class = find_classes(directory)
            self.num_imgs, self.img_id_to_path, self.img_labels = assign_img_identifier(directory, self.classes)
        else: # samples from all classes are in the same directory
            entries = sorted(entry.name for entry in os.scandir(directory))
            self.num_imgs = len(entries)
            self.img_id_to_path = []
            for i, img_name in enumerate(entries):
                self.img_id_to_path.append(img_name)
            self.img_labels = []
            label_file = open(label_file)
            line = label_file.readline()
            while line:
                self.img_labels.append(int(line))
                line = label_file.readline()

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


    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        idx = int(idx)
        img_path = self.img_id_to_path[idx]
        label = self.img_labels[idx]
        img = np.asarray(Image.open(img_path).convert("RGB"))
        return img, label



def get_ffcv_loader(dataset, nick_name, batch_size = 128,
                    num_workers = 8,
                    aug=False, scale_for_ct=False):

    if scale_for_ct: res = 64
    else: res = 224

    if aug:
        decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(0, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]
    else:
        decoder = SimpleRGBImageDecoder()
        image_pipeline: List[Operation] = [
            decoder,
            transforms.Resize(size=[res, res]),
            ToTensor(),
            ToDevice(0, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(0, non_blocking=True),
    ]


    pipelines ={
        'image': image_pipeline,
        'label': label_pipeline
    }

    cache_dir = os.path.join(root_dir, 'ffcv_cache')
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    write_path = os.path.join(cache_dir, nick_name)

    print('search for :', write_path)

    if not os.path.exists(write_path):
        print('[Fail to find %s...]' % write_path)
        print('Now, compile the ffcv-format dataset into %s' % write_path)
        # Pass a type for each data field
        writer = DatasetWriter(write_path, {
            # Tune options to optimize dataset size, throughput at train-time
            'image': RGBImageField(
                max_resolution=256,
            ),
            'label': IntField()
        })
        # Write dataset
        writer.from_indexed_dataset(dataset)
    else:
        print('Found!')


    order = OrderOption.QUASI_RANDOM
    loader = Loader(write_path,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=True,
                    drop_last=True,
                    pipelines=pipelines)

    """
    loader = Loader(write_path, batch_size=batch_size, num_workers=num_workers,
                    order=OrderOption.RANDOM, pipelines=pipelines)"""

    return loader