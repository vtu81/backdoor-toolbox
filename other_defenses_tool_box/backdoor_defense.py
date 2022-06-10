import config, os
from utils import supervisor
import torch
from torchvision import datasets, transforms
from PIL import Image

class BackdoorDefense():
    def __init__(self, args):
        self.dataset = args.dataset
        if args.dataset == 'gtsrb':
            if args.no_normalize:
                self.data_transform_aug = transforms.Compose([
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                ])
                self.data_transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                self.normalizer = transforms.Compose([])
                self.denormalizer = transforms.Compose([])
            else:
                self.data_transform_aug = transforms.Compose([
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
                self.data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
                self.normalizer = transforms.Compose([
                    transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                ])
                self.denormalizer = transforms.Compose([
                    transforms.Normalize((-0.3337 / 0.2672, -0.3064 / 0.2564, -0.3171 / 0.2629),
                                            (1.0 / 0.2672, 1.0 / 0.2564, 1.0 / 0.2629)),
                ])
            self.img_size = 32
            self.num_classes = 43
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'cifar10':
            if args.no_normalize:
                self.data_transform_aug = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor()
                ])
                self.data_transform = transforms.Compose([
                    transforms.ToTensor()
                ])
                self.normalizer = transforms.Compose([])
                self.denormalizer = transforms.Compose([])
            else:
                self.data_transform_aug = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomCrop(32, 4),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
                self.data_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
                self.normalizer = transforms.Compose([
                    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
                ])
                self.denormalizer = transforms.Compose([
                    transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
                ])            
            self.img_size = 32
            self.num_classes = 10
            self.input_channel = 3
            self.shape = torch.Size([3, 32, 32])
            self.momentum = 0.9
            self.weight_decay = 1e-4
            self.learning_rate = 0.1
        elif args.dataset == 'cifar100':
            print('<To Be Implemented> Dataset = %s' % args.dataset)
            exit(0)
        elif args.dataset == 'imagenette':
            print('<To Be Implemented> Dataset = %s' % args.dataset)
            exit(0)
        else:
            print('<Undefined> Dataset = %s' % args.dataset)
            exit(0)
        
        self.poison_type = args.poison_type
        self.poison_rate = args.poison_rate
        self.cover_rate = args.cover_rate
        self.alpha = args.alpha
        self.trigger = args.trigger
        self.target_class = config.target_class[args.dataset]
        self.device='cuda'

        self.poison_transform = supervisor.get_poison_transform(poison_type=args.poison_type, dataset_name=args.dataset,
                                                            target_class=config.target_class[args.dataset], trigger_transform=self.data_transform,
                                                            is_normalized_input=(not args.no_normalize),
                                                            alpha=args.alpha if args.test_alpha is None else args.test_alpha,
                                                            trigger_name=args.trigger)
        
        trigger_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trigger_path = os.path.join(config.triggers_dir, args.trigger)
        print('trigger_path:', trigger_path)
        self.trigger_mark = Image.open(trigger_path).convert("RGB")
        self.trigger_mark = trigger_transform(self.trigger_mark).cuda()
        
        trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % args.trigger)
        if os.path.exists(trigger_mask_path): # if there explicitly exists a trigger mask (with the same name)
            print('trigger_mask_path:', trigger_mask_path)
            self.trigger_mask = Image.open(trigger_mask_path).convert("RGB")
            self.trigger_mask = transforms.ToTensor()(self.trigger_mask)[0].cuda() # only use 1 channel
        else: # by default, all black pixels are masked with 0's (not used)
            print('No trigger mask found! By default masking all black pixels...')
            self.trigger_mask = torch.logical_or(torch.logical_or(self.trigger_mark[0] > 0, self.trigger_mark[1] > 0), self.trigger_mark[2] > 0).cuda()

        self.poison_set_dir = supervisor.get_poison_set_dir(args)
        model_path = supervisor.get_model_dir(args)
        arch = config.arch[args.dataset]
        self.model = arch(num_classes=self.num_classes)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print("Evaluating model '{}'...".format(model_path))
        else:
            print("Model '{}' not found.".format(model_path))
        
        self.model = self.model.cuda()
        self.model.eval()
        
