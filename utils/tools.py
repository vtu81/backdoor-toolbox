import  torch
from torch import nn
import  torch.nn.functional as F
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random
import numpy as np
from torchvision.utils import save_image
import config
from utils import supervisor


class IMG_Dataset(Dataset):
    def __init__(self, data_dir, label_path, transforms = None):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """
        self.dir = data_dir
        self.gt = torch.load(label_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        idx = int(idx)
        img = Image.open(os.path.join(self.dir, '%d.png' % idx))
        if self.transforms is not None:
            img = self.transforms(img)
        label = self.gt[idx]
        return img, label



def test(model, test_loader, poison_test = False, poison_transform=None, num_classes=10, source_classes=None):

    model.eval()
    clean_correct = 0
    poison_correct = 0
    non_source_classified_as_target = 0
    tot = 0
    num_non_target_class = 0
    criterion = nn.CrossEntropyLoss()
    tot_loss = 0
    poison_acc = 0

    class_dist = np.zeros((num_classes))

    with torch.no_grad():
        for data, target in test_loader:

            data, target = data.cuda(), target.cuda()
            clean_output = model(data)
            clean_pred = clean_output.argmax(dim=1)
            clean_correct += clean_pred.eq(target).sum().item()

            tot += len(target)
            this_batch_size = len(target)
            tot_loss += criterion(clean_output, target) * this_batch_size


            for bid in range(this_batch_size):
                if clean_pred[bid] == target[bid]:
                    class_dist[target[bid]] += 1

            if poison_test:
                clean_target = target
                data, target = poison_transform.transform(data, target)

                poison_output = model(data)
                poison_pred = poison_output.argmax(dim=1, keepdim=True)

                target_class = target[0].item()
                for bid in range(this_batch_size):
                    if clean_target[bid]!=target_class:
                        if source_classes is None:
                            num_non_target_class+=1
                            if poison_pred[bid] == target_class:
                                poison_correct+=1
                        else: # for source-specific attack
                            if clean_target[bid] in source_classes:
                                num_non_target_class+=1
                                if poison_pred[bid] == target_class:
                                    poison_correct+=1

                poison_acc += poison_pred.eq((clean_target.view_as(poison_pred))).sum().item()


    if poison_test:
        print('ASR: %d/%d = %.6f' % (poison_correct, num_non_target_class, poison_correct / num_non_target_class))
        print('Attack ACC: %d/%d = %.6f' % (poison_acc, tot, poison_acc/tot) )
    
    print('Clean ACC: {}/{} = {:.6f}, Loss: {}'.format(
            clean_correct, tot,
            clean_correct/tot, tot_loss/tot
    ))
    print('Class_Dist: ', class_dist)
    print("")
    return  clean_correct/tot


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_dataset(dataset, path):
    num = len(dataset)
    label_set = []

    if not os.path.exists(path):
        os.mkdir(path)

    img_dir = os.path.join(path,'data')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)


    for i in range(num):
        img, gt = dataset[i]
        img_file_name = '%d.png' % i
        img_file_path = os.path.join(img_dir, img_file_name)
        save_image(img, img_file_path)
        print('[Generate Test Set] Save %s' % img_file_path)
        label_set.append(gt)

    label_set = torch.LongTensor(label_set)
    label_path = os.path.join(path, 'labels')
    torch.save(label_set, label_path)
    print('[Generate Test Set] Save %s' % label_path)


def unpack_poisoned_train_set(args, batch_size=128, shuffle=False, data_transform=None):
    """
    Return with `poison_set_dir`, `poisoned_set_loader`, `poison_indices`, and `cover_indices` if available
    """
    if args.dataset == 'cifar10':
        if data_transform is None:
            if args.no_normalize:
                data_transform = transforms.Compose([
                        transforms.ToTensor(),
                ])
            else:
                data_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
    elif args.dataset == 'gtsrb':
        if data_transform is None:
            if args.no_normalize:
                data_transform = transforms.Compose([
                    transforms.ToTensor(),
                ])
            else:
                data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
    else: raise NotImplementedError()

    poison_set_dir = supervisor.get_poison_set_dir(args)

    poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    poisoned_set_label_path = os.path.join(poison_set_dir, 'labels')
    poison_indices_path = os.path.join(poison_set_dir, 'poison_indices')
    cover_indices_path = os.path.join(poison_set_dir, 'cover_indices') # for adaptive attacks

    poisoned_set = IMG_Dataset(data_dir=poisoned_set_img_dir,
                                label_path=poisoned_set_label_path, transforms=data_transform)

    poisoned_set_loader = torch.utils.data.DataLoader(poisoned_set, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

    poison_indices = torch.load(poison_indices_path)
    
    if args.poison_type == 'adaptive' or args.poison_type == 'TaCT':
        cover_indices = torch.load(cover_indices_path)
        return poison_set_dir, poisoned_set_loader, poison_indices, cover_indices
    
    return poison_set_dir, poisoned_set_loader, poison_indices, []
