import numpy as np
import torch
import os, sys
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch import nn
from utils import supervisor, tools
import config
import confusion_training
import random

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False, default=config.parser_default['dataset'],
                    choices=config.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str,  required=True,
        choices=config.parser_choices['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=config.parser_choices['poison_rate'],
                    default=config.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=config.parser_choices['cover_rate'],
                    default=config.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False, default=config.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float,  required=False, default=None)
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-debug_info', default=False, action='store_true')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=config.seed)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
tools.setup_seed(args.seed)
if args.trigger is None:
    args.trigger = config.trigger_default[args.poison_type]
if args.log:
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'cleanse')
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'CT_%s.out' % (supervisor.get_dir_core(args, include_poison_seed=config.record_poison_seed)))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

params = config.get_params(args)
inspection_set, clean_set = config.get_dataset(params['inspection_set_dir'], params['data_transform'], args)

debug_packet = None
if args.debug_info:
    debug_packet = config.get_packet_for_debug(params['inspection_set_dir'], params['data_transform'], params['batch_size'], args)


distilled_samples_indices, median_sample_indices = confusion_training.iterative_poison_distillation(
    inspection_set, clean_set, params, args, debug_packet)
distilled_set = torch.utils.data.Subset(inspection_set, distilled_samples_indices)


inference_model = confusion_training.generate_inference_model(
    distilled_set, clean_set, params, args, debug_packet)


print('>>> Dataset Cleanse ...')
num_classes = params['num_classes']

suspicious_indices = confusion_training.cleanser(args = args, inspection_set=inspection_set, clean_set_indices = median_sample_indices,
                               model=inference_model, num_classes=num_classes)

suspicious_indices.sort()
remain_indices = list( set(range(0,len(inspection_set))) - set(suspicious_indices) )
remain_indices.sort()

remain_dist = np.zeros(num_classes)
for temp_id in remain_indices:
    _, gt = inspection_set[temp_id]
    gt = gt.item()
    remain_dist[gt]+=1
print('remain dist : ', remain_dist)


save_path = os.path.join(params['inspection_set_dir'], 'cleansed_set_indices_seed=%d' % args.seed)
torch.save(remain_indices, save_path)
print('[Save] %s' % save_path)

if args.debug_info: # evaluate : how many poison samples are eliminated ?

    poison_indices = debug_packet['poison_indices']
    poison_indices.sort()

    true_positive = 0
    num_positive = 0
    num_negative = 0
    false_positive = 0

    tot_poison = len(poison_indices)
    num_samples = len(inspection_set)

    pt = 0
    for pid in range(num_samples):
        while pt+1 < tot_poison and poison_indices[pt] < pid: pt+=1
        if pt < tot_poison and poison_indices[pt] == pid : num_positive+=1
        else: num_negative+=1

    pt = 0
    for pid in suspicious_indices:
        while pt+1 < tot_poison and poison_indices[pt] < pid: pt+=1
        if pt < tot_poison and poison_indices[pt] == pid: true_positive+=1
        else: false_positive+=1

    tpr = 0 if num_positive == 0 else true_positive/num_positive
    print('Elimination Rate = %d/%d = %f' % (true_positive, num_positive, tpr) )
    fpr = 0 if num_negative == 0 else false_positive/num_negative
    print('Sacrifice Rate = %d/%d = %f' % (false_positive, num_negative, fpr))