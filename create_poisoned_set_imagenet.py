import os
import torch
import argparse
from utils import default_args, tools, supervisor, imagenet
import torchvision.transforms as transforms
import random
from PIL import Image
from torchvision.utils import save_image
import config

parser = argparse.ArgumentParser()
parser.add_argument('-poison_type', type=str,  required=False,
                    choices=default_args.parser_choices['poison_type'],
                    default=default_args.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-alpha', type=float,  required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
args = parser.parse_args()
args.dataset = 'imagenet'
tools.setup_seed(0)

if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]

print(f"Please notice for ImageNet, the trigger '{args.trigger}' will be resized to 256x256! Specifically:\n\
1. The **training** images are first resized to 256x256 and then randomly cropped to 224x224. \
And while poisoning the **training** set, the trigger is resized to 256x256 and then planted into the 256x256 clean images.\n\
2. The **test** images are first resized to 256x256 and then center cropped to 224x224.)\
So while poisoning the **test** inputs, the trigger is resized to 256x256, center cropped to 224x224, and then added to the 224x224 clean images")

if args.poison_type not in ['none', 'badnet', 'trojan', 'blend']:
    raise NotImplementedError('%s is not implemented on ImageNet' % args.poison_type)

if args.poison_type == 'none':
    args.poison_rate = 0

if not os.path.exists(os.path.join('poisoned_train_set', 'imagenet')):
    os.mkdir(os.path.join('poisoned_train_set', 'imagenet'))


poison_set_dir = supervisor.get_poison_set_dir(args)
if not os.path.exists(poison_set_dir):
    os.mkdir(poison_set_dir)

poison_imgs_dir = os.path.join(poison_set_dir, 'data')
if not os.path.exists(poison_imgs_dir):
    os.mkdir(poison_imgs_dir)

num_imgs = 1281167 # size of imagenet training set


# random sampling
id_set = list(range(0,num_imgs))
random.shuffle(id_set)
num_poison = int(num_imgs * args.poison_rate)
poison_indices = id_set[:num_poison]
poison_indices.sort() # increasing order


# train_set_dir = '/shadowdata/xiangyu/imagenet_256/train'
# train_set_dir = '/scratch/gpfs/DATASETS/imagenet/ilsvrc_2012_classification_localization/train'
train_set_dir = os.path.join(config.imagenet_dir, "train")

classes, class_to_idx, idx_to_class = imagenet.find_classes(train_set_dir)
num_imgs, img_id_to_path, img_labels = imagenet.assign_img_identifier(train_set_dir, classes)

transform_to_tensor = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# poison_transform = imagenet.get_poison_transform_for_imagenet(args.poison_type)
poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                    target_class=config.target_class[args.dataset], trigger_transform=transform_to_tensor,
                                                    is_normalized_input=True,
                                                    alpha=args.alpha,
                                                    trigger_name=args.trigger, args=args)


cnt = 0
tot = len(poison_indices)
print('# poison samples = %d' % tot)
for pid in poison_indices:
    cnt+=1
    ori_img = transform_to_tensor(Image.open(os.path.join(train_set_dir, img_id_to_path[pid])).convert("RGB"))
    poison_img, _ = poison_transform.transform(ori_img, torch.zeros(ori_img.shape[0]))

    cls_path = os.path.join(poison_imgs_dir, idx_to_class[img_labels[pid]])
    if not os.path.exists(cls_path):
        os.mkdir(cls_path)

    dst_path = os.path.join(poison_imgs_dir, img_id_to_path[pid])
    save_image(poison_img, dst_path)
    print('save [%d/%d]: %s' % (cnt,tot, dst_path))



poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
torch.save(poison_indices, poison_indices_path)
print('[Generate Poisoned Set] Save %s' % poison_indices_path)