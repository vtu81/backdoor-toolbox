import random
import os
import torch
import numpy as np
import PIL
from PIL import Image
from numpy.random.mtrand import poisson
from torchvision.utils import save_image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from torchvision import transforms
import cv2
from scipy import stats
from config import poison_seed

"""
Code referenced from https://github.com/THUYimingLi/BackdoorBox.
Default `ghost_rate` is set to 1 (instead of 0.49).
Default `ghost_alpha` random range is set to [0.5, 0.75] (instead of [0.15, 0.35]).
"""

def read_image(img_path, type=None):
            img = cv2.imread(img_path)
            if type is None:        
                return img
            elif isinstance(type,str) and type.upper() == "RGB":
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif isinstance(type,str) and type.upper() == "GRAY":
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                raise NotImplementedError

class poison_generator():
    
    def __init__(self, img_size, dataset, poison_rate, path, target_class=0,
                max_image_size=560, ghost_rate=1, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0

        # number of images
        self.num_img = len(dataset)

        # load reflection images
        reflection_data_dir = "data/VOCdevkit/VOC2012/JPEGImages/" # please replace this with path to your desired reflection set
        if not os.path.exists(reflection_data_dir):
            print(f"Reflection images data {reflection_data_dir} not exist! Please first download them by running the following script at 'data/':")
            print("```")
            print("wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")
            print("tar -xvf VOCtrainval_11-May-2012.tar")
            print("```")
            exit()
        reflection_image_path = os.listdir(reflection_data_dir)
        self.reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
        self.AddTriggerMixin = AddTriggerMixin(
            total_num=self.num_img,
            reflection_cadidates=self.reflection_images,
            max_image_size=max_image_size,
            ghost_rate=ghost_rate,
            alpha_b=alpha_b,
            offset=offset,
            sigma=sigma,
            ghost_alpha=ghost_alpha)

    def generate_poisoned_training_set(self):
        torch.manual_seed(poison_seed)
        random.seed(poison_seed)

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort() # increasing order

        img_set = []
        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                img = self.AddTriggerMixin._add_trigger(img * 255, i) / 255.0
                pt+=1

            # img_file_name = '%d.png' % i
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            #print('[Generate Poisoned Set] Save %s' % img_file_path)
            
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)
        
        img, gt = self.dataset[0]
        img = self.AddTriggerMixin._add_trigger(img * 255, i) / 255.0
        save_image(img, os.path.join(self.path, 'demo.png'))
        
        return img_set, poison_indices, label_set



class poison_transform():
    def __init__(self, img_size, denormalizer, normalizer, target_class=0,
                max_image_size=560, ghost_rate=1, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.):
        
        self.img_size = img_size
        self.normalizer = normalizer
        self.denormalizer = denormalizer
        self.target_class = target_class # by default : target_class = 0
        
        reflection_data_dir = "data/VOCdevkit/VOC2012/JPEGImages/" # please replace this with path to your desired reflection set            
        reflection_image_path = os.listdir(reflection_data_dir)
        self.reflection_images = [read_image(os.path.join(reflection_data_dir,img_path)) for img_path in reflection_image_path[:200]]
        
        self.max_image_size = max_image_size
        self.ghost_rate = ghost_rate
        self.alpha_b = alpha_b
        self.offset = offset
        self.sigma = sigma
        self.ghost_alpha = ghost_alpha
        
    def transform(self, data, labels):
        data = data.clone()
        labels = labels.clone()
        device = data.device
        
        # transform clean samples to poison samples
        labels[:] = self.target_class
        data = self.denormalizer(data).cpu()

        self.AddTriggerMixin = AddTriggerMixin(
            total_num=len(data),
            reflection_cadidates=self.reflection_images,
            max_image_size=self.max_image_size,
            ghost_rate=self.ghost_rate,
            alpha_b=self.alpha_b,
            offset=self.offset,
            sigma=self.sigma,
            ghost_alpha=self.ghost_alpha)

        for (i, img) in enumerate(data):
            data[i] = self.AddTriggerMixin._add_trigger(img * 255, i) / 255.0
        
        data = self.normalizer(data).to(device=device)

        # debug
        # from torchvision.utils import save_image
        # from torchvision import transforms
        # preprocess = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # reverse_preprocess = transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        # save_image(reverse_preprocess(data)[-7], 'a.png')

        return data, labels




class AddTriggerMixin(object):
    """Add reflection-based trigger to images.

    Args:
        total_num (integer): number of images in the dataset
        reflection_cadidates (List of numpy.ndarray of shape (H, W, C) or (H, W))
        max_image_size (integer): max(Height, Weight) of returned image
        ghost_rate (float): rate of ghost reflection
        alpha_b (float): the ratio of background image in blended image, alpha_b should be in $(0,1)$, set to -1 if random alpha_b is desired
        offset (tuple of 2 interger): the offset of ghost reflection in the direction of x axis and y axis, set to (0,0) if random offset is desired
        sigma (interger): the sigma of gaussian kernel, set to -1 if random sigma is desired
        ghost_alpha (interger): ghost_alpha should be in $(0,1)$, set to -1 if random ghost_alpha is desire
    """
    def __init__(self, total_num, reflection_cadidates, max_image_size=560, ghost_rate=1, alpha_b=-1., offset=(0, 0), sigma=-1, ghost_alpha=-1.):
        super(AddTriggerMixin,self).__init__()
        self.reflection_candidates = reflection_cadidates
        self.max_image_size=max_image_size
        # generate random numbers for refelection-based trigger generation and keep them fixed during training 
        self.reflection_candidates_index = np.random.randint(0,len(self.reflection_candidates),total_num)
        self.alpha_bs = 1.-np.random.uniform(0.05,0.45,total_num) if alpha_b<0 else np.zeros(total_num)+alpha_b
        self.ghost_values = (np.random.uniform(0,1,total_num) < ghost_rate)
        if offset == (0,0):
            self.offset_xs = np.random.random_integers(3,8,total_num)
            self.offset_ys = np.random.random_integers(3,8,total_num)
        else:
            self.offset_xs = np.zeros((total_num,),np.int32) + offset[0]
            self.offset_ys = np.zeros((total_num,),np.int32) + offset[1]
        self.ghost_alpha = ghost_alpha
        self.ghost_alpha_switchs = np.random.uniform(0,1,total_num)
        # self.ghost_alphas = np.random.uniform(0.15,0.5,total_num) if ghost_alpha < 0 else np.zeros(total_num)+ghost_alpha
        self.ghost_alphas = np.random.uniform(0.5,0.75,total_num) if ghost_alpha < 0 else np.zeros(total_num)+ghost_alpha
        self.sigmas = np.random.uniform(1,5,total_num) if sigma<0 else np.zeros(total_num)+sigma
        self.atts = 1.08 + np.random.random(total_num)/10.0
        self.new_ws = np.random.uniform(0,1,total_num)
        self.new_hs = np.random.uniform(0,1,total_num)

    def _add_trigger(self, sample, index):
        """Add reflection-based trigger to images.

        Args:        
            sample (torch.Tensor): shape (C,H,W),
            index (interger): index of sample in original dataset
        """
        img_b = sample.permute(1,2,0).numpy() # background
        img_r = self.reflection_candidates[self.reflection_candidates_index[index]] # reflection
        h, w, channels = img_b.shape
        if channels == 1 and img_r.shape[-1]==3: 
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)[:,:,np.newaxis]

        b = np.float32(img_b) / 255.
        r = np.float32(img_r) / 255.
        
        # convert t.shape to max_image_size's limitation
        scale_ratio = float(max(h, w)) / float(self.max_image_size)
        w, h = (self.max_image_size, int(round(h / scale_ratio))) if w > h \
            else (int(round(w / scale_ratio)), self.max_image_size)
        b = cv2.resize(b, (w, h), cv2.INTER_CUBIC)
        r = cv2.resize(r, (w, h), cv2.INTER_CUBIC)
        if channels == 1:
            b = b[:,:,np.newaxis]
            r = r[:,:,np.newaxis]
        
        alpha_b = self.alpha_bs[index]
        if self.ghost_values[index]:
            b = np.power(b, 2.2)
            r = np.power(r, 2.2)

            # generate the blended image with ghost effect
            offset = (self.offset_xs[index],self.offset_ys[index])
            r_1 = np.lib.pad(r, ((0, offset[0]), (0, offset[1]), (0, 0)),
                         'constant', constant_values=0)
            r_2 = np.lib.pad(r, ((offset[0], 0), (offset[1], 0), (0, 0)),
                         'constant', constant_values=(0, 0))
            ghost_alpha = self.ghost_alpha
            if ghost_alpha < 0:
                ghost_alpha_switch = 1 if self.ghost_alpha_switchs[index] > 0.5 else 0
                ghost_alpha = abs(ghost_alpha_switch - self.ghost_alphas[index])
            
            ghost_r = r_1 * ghost_alpha + r_2 * (1 - ghost_alpha)
            ghost_r = cv2.resize(ghost_r[offset[0]: -offset[0], offset[1]: -offset[1], :], (w, h))
            if channels==1:
                ghost_r = ghost_r[:,:,np.newaxis]
            reflection_mask = ghost_r * (1 - alpha_b)
            blended = reflection_mask + b * alpha_b
            transmission_layer = np.power(b * alpha_b, 1 / 2.2)

            ghost_r = np.power(reflection_mask, 1 / 2.2)
            ghost_r[ghost_r > 1.] = 1.
            ghost_r[ghost_r < 0.] = 0.

            blended = np.power(blended, 1 / 2.2)
            blended[blended > 1.] = 1.
            blended[blended < 0.] = 0.

            reflection_layer = np.uint8(ghost_r * 255)
            blended = np.uint8(blended * 255)
            transmission_layer = np.uint8(transmission_layer * 255)
        else:
            # generate the blended image with focal blur
            sigma = self.sigmas[index]

            b = np.power(b, 2.2)
            r = np.power(r, 2.2)

            sz = int(2 * np.ceil(2 * sigma) + 1)
            r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
            if channels==1:
                r_blur = r_blur[:,:,np.newaxis]
            blend = r_blur + b

            # get the reflection layers' proper range
            att = self.atts[index]
            for i in range(channels):
                maski = blend[:, :, i] > 1
                mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
                r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
            r_blur[r_blur >= 1] = 1
            r_blur[r_blur <= 0] = 0

            def gen_kernel(kern_len=100, nsig=1):
                """Returns a 2D Gaussian kernel array."""
                interval = (2 * nsig + 1.) / kern_len
                x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kern_len + 1)
                # get normal distribution
                kern1d = np.diff(stats.norm.cdf(x))
                kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
                kernel = kernel_raw / kernel_raw.sum()
                kernel = kernel / kernel.max()
                return kernel
            h, w = r_blur.shape[0: 2]
            new_w = int(self.new_ws[index]*(self.max_image_size - w - 10)) if w < self.max_image_size - 10 else 0
            new_h = int(self.new_hs[index]*(self.max_image_size - h - 10)) if h < self.max_image_size - 10 else 0

            g_mask = gen_kernel(self.max_image_size, 3)
            g_mask = np.dstack((g_mask, )*channels)
            alpha_r = g_mask[new_h: new_h + h, new_w: new_w + w, :] * (1. - alpha_b / 2.)

            r_blur_mask = np.multiply(r_blur, alpha_r)
            blur_r = min(1., 4 * (1 - alpha_b)) * r_blur_mask
            blend = r_blur_mask + b * alpha_b

            transmission_layer = np.power(b * alpha_b, 1 / 2.2)
            r_blur_mask = np.power(blur_r, 1 / 2.2)
            blend = np.power(blend, 1 / 2.2)
            blend[blend >= 1] = 1
            blend[blend <= 0] = 0
            blended = np.uint8(blend * 255)
        return torch.from_numpy(blended).permute(2, 0, 1)