from other_attacks_tool_box import BackdoorAttack
from utils import supervisor
from other_attacks_tool_box.tools import generate_dataloader
import config
import copy
import torch
import numpy as np
import random


class attacker:

    def __init__(self, args, mode="all2one", dithering=True, squeeze_num=8, injection_rate=0.2):
        self.args = args
        self.dataset = args.dataset
        self.num_classes = 10
        self.mode = mode
        self.dithering = dithering
        self.squeeze_num = squeeze_num
        self.injection_rate = injection_rate
        self.target_class = config.target_class[args.dataset]
        self.model = supervisor.get_arch(args)(num_classes=self.num_classes)
        self.data_transform_aug, self.data_transform, self.trigger_transform, self.normalizer, self.denormalizer = supervisor.get_transforms(
            args)
        self.train_loader = generate_dataloader(dataset=self.dataset,
                                                dataset_path=config.data_dir,
                                                batch_size=100,
                                                split='train',
                                                shuffle=False,
                                                drop_last=False,
                                                )
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='full_test',
                                               shuffle=False,
                                               drop_last=False,
                                               )
        self.optimizer = torch.optim.SGD(self.model.parameters(), 1e-2, momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [100, 200, 300, 400], 0.1)
        self.criterion_CE = torch.nn.CrossEntropyLoss()
        self.criterion_BCE = torch.nn.BCELoss()

    def back_to_img(self, data):
        return self.denormalizer(data) * 255

    def img_tensor_norm(self, data):
        return self.normalizer(data / 255.0)

    def rnd1(self, x, decimals, out):
        return np.round_(x, decimals, out)

    def floydDitherspeed(self, image, squeeze_num):
        channel, h, w = image.shape
        for y in range(h):
            for x in range(w):
                old = image[:, y, x]
                temp = np.empty_like(old).astype(np.float64)
                new = self.rnd1(old / 255.0 * (squeeze_num - 1), 0, temp) / (squeeze_num - 1) * 255
                error = old - new
                image[:, y, x] = new
                if x + 1 < w:
                    image[:, y, x + 1] += error * 0.4375
                if (y + 1 < h) and (x + 1 < w):
                    image[:, y + 1, x + 1] += error * 0.0625
                if y + 1 < h:
                    image[:, y + 1, x] += error * 0.3125
                if (x - 1 >= 0) and (y + 1 < h):
                    image[:, y + 1, x - 1] += error * 0.1875
        return image

    def attack(self):
        self.model.cuda()
        residual_list_train = []
        for _ in range(5):
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs = inputs.cuda()
                temp_negetive = self.back_to_img(inputs)

                temp_negetive_modified = copy.deepcopy(temp_negetive)
                if self.dithering:
                    for i in range(temp_negetive_modified.shape[0]):
                        temp_negetive_modified[i, :, :, :] = torch.round(torch.from_numpy(
                            self.floydDitherspeed(temp_negetive_modified[i].detach().cpu().numpy(),
                                                  float(self.squeeze_num))))
                else:
                    temp_negetive_modified = torch.round(temp_negetive_modified / 255.0 * (self.squeeze_num - 1)) / (
                            self.squeeze_num - 1) * 255

                residual = temp_negetive_modified - temp_negetive
                for i in range(residual.shape[0]):
                    residual_list_train.append(residual[i].unsqueeze(0).cuda())

        for epoch in range(1000):
            self.model.train()
            total_loss_ce = 0
            total_sample = 0
            total_clean = 0
            total_bd = 0
            total_clean_correct = 0
            total_bd_correct = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                inputs, targets = inputs.cuda(), targets.cuda()
                bs = inputs.shape[0]
                num_bd = int(bs * 0.2)
                num_neg = int(bs * 0.2)
                inputs_bd = self.back_to_img(inputs[:num_bd])
                if self.dithering:
                    for i in range(inputs_bd.shape[0]):
                        inputs_bd[i, :, :, :] = torch.round(torch.from_numpy(
                            self.floydDitherspeed(inputs_bd[i].detach().cpu().numpy(), float(self.squeeze_num))).cuda())
                else:
                    inputs_bd = torch.round(inputs_bd / 255.0 * (self.squeeze_num - 1)) / (self.squeeze_num - 1) * 255

                inputs_bd = self.back_to_img(inputs_bd)

                if self.mode == "all2one":
                    targets_bd = torch.ones_like(targets[:num_bd]) * self.target_class
                if self.mode == "all2all":
                    targets_bd = torch.remainder(targets[:num_bd] + 1, self.num_classes)

                inputs_negative = self.back_to_img(inputs[num_bd: (num_bd + num_neg)]) + torch.cat(
                    random.sample(residual_list_train, num_neg), dim=0)
                inputs_negative = torch.clamp(inputs_negative, 0, 255)
                inputs_negative = self.img_tensor_norm(inputs_negative)

                total_inputs = torch.cat([inputs_bd, inputs_negative, inputs[(num_bd + num_neg):]], dim=0)
                total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

                total_inputs = self.img_tensor_norm(total_inputs)
                total_preds = self.model(total_inputs)
                loss_ce = self.criterion_CE(total_preds, total_targets)
                loss = loss_ce
                loss.backward()
                self.optimizer.step()

                total_bd += num_bd
                total_sample += bs
                total_loss_ce += loss_ce.detach()
                total_clean += bs - num_bd - num_neg
                total_clean_correct += torch.sum(
                    torch.argmax(total_preds[(num_bd + num_neg):], dim=1) == total_targets[(num_bd + num_neg):]
                )
                total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
                avg_acc_bd = total_bd_correct * 100.0 / total_bd
                avg_acc_clean = total_clean_correct * 100.0 / total_clean

                print("Epoch {} - Batch {}: Clean - {}, Backdoor - {}".format(epoch + 1, batch_idx, avg_acc_clean,
                                                                              avg_acc_bd))

        self.scheduler.step()
