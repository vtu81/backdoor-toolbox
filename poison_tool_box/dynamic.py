import os
import torch
import random
from torchvision.utils import save_image
from torch import nn
from torchvision import transforms

class poison_generator():

    def __init__(self, ckpt_path, channel_init, steps, input_channel, normalizer, denormalizer,
                 dataset, poison_rate, path, target_class=0, cuda_devices='0'):

        os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % cuda_devices

        # official pretrained pattern & mask generator model
        state_dict = torch.load(ckpt_path)

        self.dataset = dataset
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class  # by default : target_class = 0
        self.denormalizer = denormalizer
        self.normalizer = normalizer

        netG = Generator(channel_init=channel_init, steps=steps, input_channel=input_channel,
                         normalizer=normalizer, denormalizer=denormalizer)
        netG.load_state_dict(state_dict["netG"])
        netG.cuda()
        netG.eval()
        netG.requires_grad_(False)
        self.pattern_generator = netG

        netM = Generator(channel_init=channel_init, steps=steps, input_channel=input_channel,
                         normalizer=normalizer, denormalizer=denormalizer, out_channels=1)
        netM.load_state_dict(state_dict["netM"])
        netM.cuda()
        netM.eval()
        netM.requires_grad_(False)
        self.mask_generator = netM

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):

        # random sampling
        id_set = list(range(0, self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()  # increasing order

        img_set = []
        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class

                inputs = img.unsqueeze(dim=0).cuda()
                pattern = self.pattern_generator(inputs)
                if self.normalizer is not None:
                    pattern = self.pattern_generator.normalize_pattern(pattern)
                masks_output = self.mask_generator.threshold(self.mask_generator(inputs))
                bd_inputs = inputs + (pattern - inputs) * masks_output
                img = bd_inputs.detach().squeeze().cpu()
                pt += 1

            if self.denormalizer is not None:
                img = self.denormalizer(img)

            # img_file_name = '%d.png' % i
            # img_file_path = os.path.join(self.path, img_file_name)
            # save_image(img, img_file_path)
            #print('[Generate Poisoned Set] Save %s' % img_file_path)
            
            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        return img_set, poison_indices, label_set


class poison_transform():
    def __init__(self, ckpt_path, channel_init, steps, input_channel, normalizer, denormalizer,
                 target_class=0, has_normalized=False, require_normalization=True):

        # marker : are input data normalized?
        self.has_normalized = has_normalized
        self.require_normalization = require_normalization

        # official pretrained pattern & mask generator model
        state_dict = torch.load(ckpt_path)
        self.target_class = target_class  # by default : target_class = 0
        self.denormalizer = denormalizer
        self.normalizer = normalizer

        netG = Generator(channel_init=channel_init, steps=steps, input_channel=input_channel,
                         normalizer=normalizer, denormalizer=denormalizer)
        netG.load_state_dict(state_dict["netG"])
        netG = netG.cuda()
        netG.eval()
        netG.requires_grad_(False)
        self.pattern_generator = netG

        netM = Generator(channel_init=channel_init, steps=steps, input_channel=input_channel,
                         normalizer=normalizer, denormalizer=denormalizer, out_channels=1)
        netM.load_state_dict(state_dict["netM"])
        netM = netM.cuda()
        netM.eval()
        netM.requires_grad_(False)
        self.mask_generator = netM

    def transform(self, data, labels):

        labels = labels.clone()
        data = data.clone()

        if (not self.has_normalized) and self.require_normalization:
            data = self.normalizer(data)
        elif self.has_normalized and (not self.require_normalization):
            data = self.denormalizer(data)

        # transform clean samples to poison samples
        labels[:] = self.target_class
        
        pattern_generator = self.pattern_generator.to(data.device)
        mask_generator = self.mask_generator.to(data.device)

        pattern = pattern_generator(data)
        pattern = pattern_generator.normalize_pattern(pattern)
        masks_output = mask_generator.threshold(mask_generator(data))
        bd_data = data + (pattern - data) * masks_output

        if (not self.has_normalized) and self.require_normalization:
            bd_data = self.denormalizer(bd_data)
        elif self.has_normalized and (not self.require_normalization):
            bd_data = self.normalizer(bd_data)

        return bd_data, labels


"""
Dynamic Trigger Generator adapted from the official implementation 
"""


class Generator(nn.Sequential):
    def __init__(self, channel_init, steps, input_channel, normalizer,
                 denormalizer, out_channels=None):

        super(Generator, self).__init__()

        channel_current = input_channel
        channel_next = channel_init

        for step in range(steps):
            self.add_module("convblock_down_{}".format(2 * step), Conv2dBlock(channel_current, channel_next))
            self.add_module("convblock_down_{}".format(2 * step + 1), Conv2dBlock(channel_next, channel_next))
            self.add_module("downsample_{}".format(step), DownSampleBlock())
            if step < steps - 1:
                channel_current = channel_next
                channel_next *= 2

        self.add_module("convblock_middle", Conv2dBlock(channel_next, channel_next))

        channel_current = channel_next
        channel_next = channel_current // 2
        for step in range(steps):
            self.add_module("upsample_{}".format(step), UpSampleBlock())
            self.add_module("convblock_up_{}".format(2 * step), Conv2dBlock(channel_current, channel_current))
            if step == steps - 1:
                self.add_module(
                    "convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next, relu=False)
                )
            else:
                self.add_module("convblock_up_{}".format(2 * step + 1), Conv2dBlock(channel_current, channel_next))
            channel_current = channel_next
            channel_next = channel_next // 2
            if step == steps - 2:
                if out_channels is None:
                    channel_next = input_channel
                else:
                    channel_next = out_channels

        self._EPSILON = 1e-7
        self._normalizer = normalizer
        self._denormalizer = denormalizer

    def forward(self, x):
        for module in self.children():
            x = module(x)
        x = nn.Tanh()(x) / (2 + self._EPSILON) + 0.5
        return x

    def normalize_pattern(self, x):
        if self._normalizer:
            x = self._normalizer(x)
        return x

    def denormalize_pattern(self, x):
        if self._denormalizer:
            x = self._denormalizer(x)
        return x

    def threshold(self, x):
        return nn.Tanh()(x * 20 - 10) / (2 + self._EPSILON) + 0.5


class Conv2dBlock(nn.Module):
    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, ker_size, stride, padding)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05, affine=True)
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, ker_size=(2, 2), stride=2, dilation=(1, 1), ceil_mode=False, p=0.0):
        super(DownSampleBlock, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=ker_size, stride=stride, dilation=dilation, ceil_mode=ceil_mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(self, scale_factor=(2, 2), mode="bilinear", p=0.0):
        super(UpSampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x



