import torch
from other_defenses_tool_box import NC, STRIP, FP, ABL, NAD
import argparse, config, os, sys
from utils import supervisor, tools

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=config.parser_default['dataset'],
                    choices=config.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str,  required=False,
                    choices=config.parser_choices['poison_type'],
                    default=config.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float,  required=False,
                    choices=config.parser_choices['poison_rate'],
                    default=config.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float,  required=False,
                    choices=config.parser_choices['cover_rate'],
                    default=config.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float,  required=False,
                    default=config.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float,  required=False, default=None)
parser.add_argument('-trigger', type=str,  required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-model', type=str, required=False, default=None)
parser.add_argument('-model_path', required=False, default=None)
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-defense', type=str, required=True,
                    choices=['ABL', 'NC', 'STRIP', 'FP', 'NAD'],)
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=config.seed)

args = parser.parse_args()

if args.trigger is None:
    args.trigger = config.trigger_default[args.poison_type]

tools.setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
if args.log:
    out_path = 'other_defenses_tool_box/logs'
    # if not os.path.exists(out_path): os.mkdir(out_path)
    # out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    # if not os.path.exists(out_path): os.mkdir(out_path)
    # out_path = os.path.join(out_path, 'other_defenses_tool_box')
    # if not os.path.exists(out_path): os.mkdir(out_path)
    if args.defense == 'ABL':
        out_path = os.path.join(out_path, '%s_%s_seed=%s.out' % (args.defense, supervisor.get_dir_core(args, include_model_name=False, include_poison_seed=config.record_poison_seed), args.seed))
        # out_path = os.path.join(out_path, '%s_%s.out' % (args.defense, supervisor.get_dir_core(args, include_model_name=False, include_poison_seed=config.record_poison_seed)))
    else:
        out_path = os.path.join(out_path, '%s_%s.out' % (args.defense, supervisor.get_dir_core(args, include_model_name=True, include_poison_seed=config.record_poison_seed)))
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr


if args.defense == 'NC':
    defense = NC(
        args,
        epoch=30,
        batch_size=32,
        init_cost=1e-3,
        patience=5,
        attack_succ_threshold=0.99,
        )
    defense.detect()
elif args.defense == 'STRIP':
    defense = STRIP(
        args,
        strip_alpha=1.0,
        N=100,
        defense_fpr=0.1,
        batch_size=128,
        )
    defense.detect()
elif args.defense == 'FP':
    defense = FP(
        args,
        prune_ratio=0.8,
        finetune_epoch=20,
        max_allowed_acc_drop=0.1,
    )
    defense.detect()
elif args.defense == 'ABL':
    defense = ABL(
        args,
        isolation_epochs=20,
        isolation_ratio=0.01,
        gradient_ascent_type='Flooding',
        gamma=0.5,
        flooding=0.5,
        do_isolate=True,
        finetuning_ascent_model=True,
        finetuning_epochs=60,
        unlearning_epochs=5,
        do_unlearn=True
    )
    defense.detect()
elif args.defense == 'NAD':
    defense = NAD(
        args,
        teacher_epochs=10,
        erase_epochs=20
    )
    defense.detect()
else: raise NotImplementedError()