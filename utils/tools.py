import  torch, torchvision
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
from tqdm import tqdm

class IMG_Dataset(Dataset):
    def __init__(self, data_dir, label_path, transforms = None, num_classes = 10, shift = False, random_labels = False,
                 fixed_label = None):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """
        self.dir = data_dir
        self.img_set = None
        if 'data' not in self.dir: # if new version
            self.img_set = torch.load(data_dir)
        self.gt = torch.load(label_path)
        self.transforms = transforms
        if 'data' not in self.dir: # if new version, remove ToTensor() from the transform list
            self.transforms = []
            for t in transforms.transforms:
                if not isinstance(t, torchvision.transforms.ToTensor):
                    self.transforms.append(t)
            self.transforms = torchvision.transforms.Compose(self.transforms)

        self.num_classes = num_classes
        self.shift = shift
        self.random_labels = random_labels
        self.fixed_label = fixed_label

        if self.fixed_label is not None:
            self.fixed_label = torch.tensor(self.fixed_label, dtype=torch.long)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, idx):
        idx = int(idx)
        
        if self.img_set is not None: # if new version
            img = self.img_set[idx]
        else: # if old version
            img = Image.open(os.path.join(self.dir, '%d.png' % idx))
        
        if self.transforms is not None:
            img = self.transforms(img)

        if self.random_labels:
            label = torch.randint(self.num_classes,(1,))[0]
        else:
            label = self.gt[idx]
            if self.shift:
                label = (label+1) % self.num_classes

        if self.fixed_label is not None:
            label = self.fixed_label

        return img, label


class EMBER_Dataset(Dataset):
    def __init__(self, x_path, y_path, normalizer = None, inverse=False):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """

        self.inverse = inverse

        self.x = np.load(x_path)

        if normalizer is None:
            from sklearn.preprocessing import StandardScaler
            self.normal = StandardScaler()
            self.normal.fit(self.x)
        else:
            self.normal = normalizer

        self.x = self.normal.transform(self.x)
        self.x = torch.FloatTensor(self.x)

        if y_path is not None:
            self.y = np.load(y_path)
            self.y = torch.FloatTensor(self.y)
        else:
            self.y = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        idx = int(idx)
        x = self.x[idx].clone()

        if self.y is not None:
            label = self.y[idx]
            if self.inverse:
                label = (label+1) if label == 0 else (label-1)
            return x, label
        else:
            return x



class EMBER_Dataset_norm(Dataset):
    def __init__(self, x_path, y_path, sts, inverse=False):
        """
        Args:
            data_dir: directory of the data
            label_path: path to data labels
            transforms: image transformation to be applied
        """

        self.inverse = inverse
        self.x = np.load(x_path)

        self.x = (self.x - sts[0])/sts[1]

        self.x = torch.FloatTensor(self.x)

        if y_path is not None:
            self.y = np.load(y_path)
            self.y = torch.FloatTensor(self.y)
        else:
            self.y = None

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        idx = int(idx)
        x = self.x[idx].clone()

        if self.y is not None:
            label = self.y[idx]
            if self.inverse:
                label = (label+1) if label == 0 else (label-1)
            return x, label
        else:
            return x


def test(model, test_loader, poison_test = False, poison_transform=None, num_classes=10, source_classes=None, all_to_all = False):

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
        for data, target in tqdm(test_loader):

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


                if not all_to_all:

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

                else:

                    for bid in range(this_batch_size):
                        num_non_target_class += 1
                        if poison_pred[bid] == target[bid]:
                            poison_correct += 1

                poison_acc += poison_pred.eq((clean_target.view_as(poison_pred))).sum().item()
    
    print('Clean ACC: {}/{} = {:.6f}, Loss: {}'.format(
            clean_correct, tot,
            clean_correct/tot, tot_loss/tot
    ))
    if poison_test:
        print('ASR: %d/%d = %.6f' % (poison_correct, num_non_target_class, poison_correct / num_non_target_class))
        # print('Attack ACC: %d/%d = %.6f' % (poison_acc, tot, poison_acc/tot) )
    # print('Class_Dist: ', class_dist)
    print("")
    
    if poison_test:
        return clean_correct/tot, poison_correct / num_non_target_class
    return clean_correct/tot, None



def test_imagenet(model, test_loader, test_backdoor_loader=None):

    model.eval()
    clean_top1 = 0
    clean_top5 = 0
    tot = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader):

            data, target = data.cuda(), target.cuda()
            clean_output = model(data)
            _, clean_pred = torch.topk(clean_output, 5, dim=1)

            this_batch_size = len(target)
            for i in range(this_batch_size):
                if clean_pred[i][0] == target[i]:
                    clean_top1 += 1
                if target[i] in clean_pred[i]:
                    clean_top5 += 1

            tot += this_batch_size

    print('<clean accuracy> top1: %d/%d = %f; top5: %d/%d = %f' % (clean_top1,tot,clean_top1/tot,
                                                                   clean_top5,tot,clean_top5/tot))
    
    clean_top1_acc = clean_top1/tot

    if test_backdoor_loader is None: return

    model.eval()
    adv_top1 = 0
    adv_top5 = 0
    tot = 0

    with torch.no_grad():

        with torch.no_grad():
            for data, target in tqdm(test_backdoor_loader):

                data, target = data.cuda(), target.cuda()
                adv_output = model(data)
                _, adv_pred = torch.topk(adv_output, 5, dim=1)

                this_batch_size = len(target)


                for i in range(this_batch_size):
                    if adv_pred[i][0] == target[i]:
                        adv_top1 += 1
                    if target[i] in adv_pred[i]:
                        adv_top5 += 1

                tot += this_batch_size

    print('<asr> top1: %d/%d = %f; top5: %d/%d = %f' % (adv_top1, tot, adv_top1 / tot,
                                                                       adv_top5, tot, adv_top5 / tot))

    adv_top1_acc = adv_top1/tot
    
    return clean_top1_acc, adv_top1_acc

def test_ember(model, test_loader, backdoor_test_loader):
    model.eval()
    clean_correct = 0
    tot = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            clean_output = model(data)
            clean_pred = (clean_output >= 0.5).long()
            clean_correct += clean_pred.eq(target).sum().item()
            tot += len(target)

    print('<clean accuracy> %d/%d = %f' % (clean_correct, tot, clean_correct/tot) )

    adv_correct = 0
    tot = 0
    with torch.no_grad():
        for data in backdoor_test_loader:
            data = data.cuda()
            adv_output = model(data)
            adv_correct += (adv_output>=0.5).sum()
            tot += data.shape[0]

    adv_wrong = tot - adv_correct
    print('<asr> %d/%d = %f' % (adv_wrong, tot, adv_wrong/tot))
    return

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.use_deterministic_algorithms(True) # for pytorch >= 1.8
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)

def worker_init(worked_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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
    if data_transform is None:
        data_transform_aug, data_transform, trigger_transform, normalizer, denormalizer = supervisor.get_transforms(args)

    poison_set_dir = supervisor.get_poison_set_dir(args)

    if os.path.exists(os.path.join(poison_set_dir, 'data')): # if old version
        poisoned_set_img_dir = os.path.join(poison_set_dir, 'data')
    if os.path.exists(os.path.join(poison_set_dir, 'imgs')): # if new version
        poisoned_set_img_dir = os.path.join(poison_set_dir, 'imgs')
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
