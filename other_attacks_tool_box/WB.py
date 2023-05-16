import pdb

from other_attacks_tool_box import BackdoorAttack
from utils import supervisor
from other_attacks_tool_box.tools import generate_dataloader
from utils.unet import UNet
import config
import torch
from utils.tools import test
import os


class attacker(BackdoorAttack):

    def __init__(self, args, mode="all2one", alpha=0.8, beta=0.2):
        super().__init__(args)
        self.args = args
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        if args.dataset == 'cifar10':
            self.num_classes = 10
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.epochs = 100
            self.alternate_train_epochs = 20
            self.milestones = torch.tensor([50, 75])
            self.learning_rate = 0.01
            self.batch_size = 128
        else:
            raise NotImplementedError()

        self.train_loader = generate_dataloader(dataset=self.dataset,
                                                dataset_path=config.data_dir,
                                                batch_size=self.batch_size,
                                                split='train',
                                                shuffle=True,
                                                drop_last=False,
                                                data_transform=self.data_transform_aug,
                                                )
        self.test_loader = generate_dataloader(dataset=self.dataset,
                                               dataset_path=config.data_dir,
                                               batch_size=100,
                                               split='full_test',
                                               shuffle=False,
                                               drop_last=False,
                                               data_transform=self.data_transform,
                                               )

        self.optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

        self.poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                                target_class=config.target_class[args.dataset],
                                                                trigger_transform=self.data_transform,
                                                                is_normalized_input=True,
                                                                args=args)
        self.criterion_CE = torch.nn.CrossEntropyLoss()
        self.folder_path = 'other_attacks_tool_box/results/bpp'
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

    def attack(self):
        self.model.cuda()

        # training the model and the trigger generator function
        for epoch in range(self.epochs):
            if epoch < self.alternate_train_epochs:
                # training the cls model
                if not epoch % 2:
                    self.model.train()
                    for idx, (inputs, targets) in enumerate(self.train_loader):
                        self.optimizer.zero_grad()
                        inputs, targets = inputs.cuda(), targets.cuda()
                        with torch.no_grad():
                            transformed_inputs, transformed_targets = self.poison_transform.transform(inputs, targets)
                        output, clean_feature = self.model(inputs, return_hidden=True)
                        transformed_output, transformed_feature = self.model(transformed_inputs, return_hidden=True)
                        loss_normal = self.criterion_CE(output, targets)
                        loss_poison = self.criterion_CE(transformed_output, transformed_targets)
                        loss = self.alpha * loss_normal + self.beta * loss_poison
                        if not idx % 10:
                            print(
                                "Alternative Truing (model) ---- Epoch {} - Step {}: Loss --- {}".format(epoch, idx,
                                                                                                         loss))
                        loss.backward()
                        self.optimizer.step()
                else:
                    self.poison_transform.update(self.train_loader, self.model, epoch)
            else:
                self.model.train()
                for idx, (inputs, targets) in enumerate(self.train_loader):
                    self.optimizer.zero_grad()
                    inputs, targets = inputs.cuda(), targets.cuda()
                    with torch.no_grad():
                        transformed_inputs, transformed_targets = self.poison_transform.transform(inputs, targets)
                    output, clean_feature = self.model(inputs, return_hidden=True)
                    transformed_output, transformed_feature = self.model(transformed_inputs, return_hidden=True)
                    loss_normal = self.criterion_CE(output, targets)
                    loss_poison = self.criterion_CE(transformed_output, transformed_targets)
                    loss = self.alpha * loss_normal + self.beta * loss_poison
                    if not idx % 10:
                        print("Inject Backdoor ---- Epoch {} - Step {}: Loss --- {}".format(epoch, idx, loss))
                    loss.backward()
                    self.optimizer.step()

            # test the ASR
            print("In epoch {}  ---  The ASR ---".format(epoch))
            test(self.model, self.test_loader, poison_test=True, poison_transform=self.poison_transform)


class poison_transform:
    def __init__(self, mode="all2one", num_classes=10, target_class=0):
        self.mode = mode
        self.num_classes = num_classes
        self.target_class = target_class  # by default : target_class = 0
        self.transform_function = UNet(3).cuda()
        self.optimizer = torch.optim.SGD(self.transform_function.parameters(), 0.01)
        self.criterion_CE = torch.nn.CrossEntropyLoss()

    def transform(self, data, labels):
        self.transform_function.eval()
        data = data.clone()
        labels = labels.clone()

        labels[:] = self.target_class

        if self.mode == "all2one":
            labels = torch.ones_like(labels) * self.target_class
        if self.mode == "all2all":
            labels = torch.remainder(labels + 1, self.num_classes)

        data = self.transform_function(data)

        return data, labels

    def DSWD_dis(self, clean_feat, poi_feat, weight):
        clean_feat = clean_feat.transpose(0, 1)
        poi_feat = poi_feat.transpose(0, 1)
        proj_clean_feat = weight.mm(clean_feat)
        proj_poi_feat = weight.mm(poi_feat)
        class_num = proj_clean_feat.size(0)
        dis = []
        for i in range(class_num):
            proj_clean_tmp, _ = torch.sort(proj_clean_feat[i, :])
            proj_poi_tmp, _ = torch.sort(proj_poi_feat[i, :])
            d = torch.abs(proj_clean_tmp - proj_poi_tmp)
            dis.append(torch.mean(d))
        dswd = torch.mean(torch.stack(dis))
        return dswd

    def update(self, train_loader, cls_model, epoch):
        cls_model.eval()
        self.transform_function.train()
        for idx, (inputs, targets) in enumerate(train_loader):
            self.optimizer.zero_grad()
            inputs, targets = inputs.cuda(), targets.cuda()
            transformed_inputs, transformed_targets = self.transform(inputs, targets)
            output, clean_feature = cls_model(inputs, return_hidden=True)
            transformed_output, transformed_feature = cls_model(transformed_inputs, return_hidden=True)
            weight_tensor = cls_model.state_dict()['module.linear.weight']
            loss_DSWD = self.DSWD_dis(clean_feature, transformed_feature, weight_tensor)
            loss_poison = self.criterion_CE(transformed_output, transformed_targets)
            loss = loss_poison + loss_DSWD
            if not idx % 10:
                print("Alternative Truing (Trigger) ---- Epoch {} - Step {}: Loss --- {}".format(epoch, idx, loss))
            loss.backward()
            self.optimizer.step()
