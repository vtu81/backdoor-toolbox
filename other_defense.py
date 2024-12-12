import torch
import argparse, config, os, sys
from utils import supervisor, tools, default_args
import time

parser = argparse.ArgumentParser()
parser.add_argument('-dataset', type=str, required=False,
                    default=default_args.parser_default['dataset'],
                    choices=default_args.parser_choices['dataset'])
parser.add_argument('-poison_type', type=str, required=False,
                    choices=default_args.parser_choices['poison_type'],
                    default=default_args.parser_default['poison_type'])
parser.add_argument('-poison_rate', type=float, required=False,
                    choices=default_args.parser_choices['poison_rate'],
                    default=default_args.parser_default['poison_rate'])
parser.add_argument('-cover_rate', type=float, required=False,
                    choices=default_args.parser_choices['cover_rate'],
                    default=default_args.parser_default['cover_rate'])
parser.add_argument('-alpha', type=float, required=False,
                    default=default_args.parser_default['alpha'])
parser.add_argument('-test_alpha', type=float, required=False, default=None)
parser.add_argument('-trigger', type=str, required=False,
                    default=None)
parser.add_argument('-no_aug', default=False, action='store_true')
parser.add_argument('-noisy_test', default=False, action='store_true')
parser.add_argument('-model', type=str, required=False, default=None)
parser.add_argument('-model_path', required=False, default=None)
parser.add_argument('-no_normalize', default=False, action='store_true')
parser.add_argument('-defense', type=str, required=True,
                    choices=default_args.parser_choices['defense'])
parser.add_argument('-devices', type=str, default='0')
parser.add_argument('-log', default=False, action='store_true')
parser.add_argument('-seed', type=int, required=False, default=default_args.seed)

args = parser.parse_args()

if args.trigger is None:
    args.trigger = config.trigger_default[args.dataset][args.poison_type]

# tools.setup_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "%s" % args.devices
if args.log:
    # out_path = 'other_defenses_tool_box/logs'
    # if not os.path.exists(out_path): os.mkdir(out_path)
    # out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    # if not os.path.exists(out_path): os.mkdir(out_path)
    # if args.defense == 'ABL':
    #     out_path = os.path.join(out_path, '%s_%s_seed=%s.out' % (args.defense, supervisor.get_dir_core(args, include_model_name=False, include_poison_seed=config.record_poison_seed), args.seed))
    #     # out_path = os.path.join(out_path, '%s_%s.out' % (args.defense, supervisor.get_dir_core(args, include_model_name=False, include_poison_seed=config.record_poison_seed)))
    # else:
    #     out_path = os.path.join(out_path, '%s_%s.out' % (args.defense, supervisor.get_dir_core(args, include_model_name=True, include_poison_seed=config.record_poison_seed)))
    out_path = 'logs'
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, '%s_seed=%s' % (args.dataset, args.seed))
    if not os.path.exists(out_path): os.mkdir(out_path)
    out_path = os.path.join(out_path, 'other_defense')
    if not os.path.exists(out_path): os.mkdir(out_path)
    if args.noisy_test:
        out_path = os.path.join(out_path, '%s_noisy_test_%s.out' % (args.defense,
                                                     supervisor.get_dir_core(args, include_model_name=True,
                                                                             include_poison_seed=config.record_poison_seed)))
    else:
        out_path = os.path.join(out_path, '%s_%s.out' % (args.defense,
                                                     supervisor.get_dir_core(args, include_model_name=True,
                                                                             include_poison_seed=config.record_poison_seed)))
    # fout = open(out_path, 'w')
    fout = open(out_path, 'w')
    ferr = open('/dev/null', 'a')
    sys.stdout = fout
    sys.stderr = ferr

start_time = time.perf_counter()

if args.defense == 'NC':
    from other_defenses_tool_box.neural_cleanse import NC
    defense = NC(
        args,
        epoch=30,
        batch_size=32,
        init_cost=1e-3,
        patience=5,
        attack_succ_threshold=0.99,
        oracle=False,
    )
    defense.detect()
    
elif args.defense == 'BTIDBFU':
    from other_defenses_tool_box.btidbfu import BTIDBFU
    defense = BTIDBFU(
        args,
    )
    defense.detect()
    
elif args.defense == 'AC':
    from other_defenses_tool_box.activation_clustering import AC
    defense = AC(
        args,
    )
    defense.detect(noisy_test=args.noisy_test)
elif args.defense == 'STRIP':
    from other_defenses_tool_box.strip import STRIP
    defense = STRIP(
        args,
        strip_alpha=1.0,
        N=100,
        defense_fpr=0.1,
        batch_size=128,
    )
    defense.detect(noisy_test=args.noisy_test)
elif args.defense == 'FP':
    from other_defenses_tool_box.fine_pruning import FP
    if args.dataset == 'cifar10':
        defense = FP(
            args,
            prune_ratio=0.99,
            finetune_epoch=100 if args.poison_type != 'SRA' else 50,
            max_allowed_acc_drop=0.1,
        )
    elif args.dataset == 'gtsrb':
        defense = FP(
            args,
            prune_ratio=0.75,
            finetune_epoch=100,
            max_allowed_acc_drop=0.1,
        )
    else:
        raise NotImplementedError()
    defense.detect()
elif args.defense == 'ABL':
    from other_defenses_tool_box.anti_backdoor_learning import ABL
    if args.dataset == 'cifar10':
        defense = ABL(
            args,
            isolation_epochs=15,
            isolation_ratio=0.001,
            # gradient_ascent_type='LGA',
            gradient_ascent_type='Flooding',
            gamma=0.01,
            flooding=0.3,
            do_isolate=True,
            finetuning_ascent_model=False,
            finetuning_epochs=60,
            unlearning_epochs=10,
            lr_unlearning=2e-2,
            do_unlearn=True,
        )
        defense.detect()
    elif args.dataset == 'gtsrb':
        defense = ABL(
            args,
            isolation_epochs=5,
            isolation_ratio=0.005,
            # gradient_ascent_type='LGA',
            gradient_ascent_type='Flooding',
            gamma=0.1,
            flooding=0.03,
            do_isolate=True,
            finetuning_ascent_model=True,
            finetuning_epochs=10,

            # # For 0.001 isolation rate
            # unlearning_epochs=10,
            # lr_unlearning=1e-3,
            # do_unlearn=True,

            # For 0.003 isolation rate
            unlearning_epochs=5,
            lr_unlearning=5e-4,
            do_unlearn=True,

            # # For 0.005 isolation rate
            # unlearning_epochs=5,
            # lr_unlearning=1e-3,
            # do_unlearn=True,
        )
        defense.detect()
elif args.defense == 'NAD':
    from other_defenses_tool_box.neural_attention_distillation import NAD
    defense = NAD(
        args,
        teacher_epochs=10,
        erase_epochs=20
    )
    defense.detect()
elif args.defense == 'SentiNet':
    from other_defenses_tool_box.sentinet import SentiNet
    defense = SentiNet(
        args,
        defense_fpr=0.1,
        N=100,
    )
    defense.detect()
elif args.defense == 'ScaleUp':
    from other_defenses_tool_box.scale_up import ScaleUp
    defense = ScaleUp(args, with_clean_data=False)
    defense.detect(noisy_test=args.noisy_test)
elif  args.defense == 'IBD_PSC':
    from other_defenses_tool_box.IBD_PSC import IBD_PSC
    defense = IBD_PSC(args)
    # defense.detect()
    defense.test()

elif args.defense == "SEAM":
    from other_defenses_tool_box.SEAM import SEAM
    defense = SEAM(args)
    defense.detect()
elif args.defense == "SFT":
    from other_defenses_tool_box.super_finetuning import SFT
    if args.dataset == 'cifar10':
        defense = SFT(args, lr_base=3e-2, lr_max1=2.5, lr_max2=0.05)
    elif args.dataset == 'gtsrb':
        defense = SFT(args, lr_base=3e-3, lr_max1=0.25, lr_max2=0.005)
    defense.detect()
elif args.defense == 'NONE':
    from other_defenses_tool_box.NONE import NONE
    # if args.dataset == 'cifar10':
    defense = NONE(args, none_lr=1e-2, max_reset_fraction=0.03, epoch_num_1=200, epoch_num_2=40)
    defense.detect()
elif args.defense == 'Frequency':
    from other_defenses_tool_box.frequency import Frequency
    defense = Frequency(args)
    defense.detect(noisy_test=args.noisy_test)
elif args.defense == 'moth':
    from other_defenses_tool_box.moth import moth
    if args.poison_type == 'SRA':
        defense = moth(args, lr=0.0001)
    elif args.dataset == 'gtsrb':
        defense = moth(args, lr=0.00001)
    else: defense = moth(args, lr=0.001)
    defense.detect()
elif args.defense == 'IBAU':
    from other_defenses_tool_box.IBAU import IBAU
    if args.dataset == 'cifar10':
        # defense = IBAU(args, optim='SGD', lr=0.07, n_rounds=3, K=5)
        defense = IBAU(args, optim='Adam', lr=0.0005, n_rounds=3, K=5)
    else: raise NotImplementedError()
    defense.detect()
elif args.defense == 'ANP':
    from other_defenses_tool_box.ANP import ANP
    if args.dataset == 'cifar10':
        defense = ANP(args, lr=0.2, anp_eps=0.4, anp_steps=1, anp_alpha=0.2, nb_iter=2000, print_every=500,
                      pruning_by='threshold', pruning_max=0.90, pruning_step=0.05, max_CA_drop=0.1)
    else: raise NotImplementedError()
    defense.detect()
elif args.defense == 'AWM':
    from other_defenses_tool_box.AWM import AWM
    if args.dataset == 'cifar10':
        defense = AWM(args, lr1=1e-3, lr2=1e-2, outer=20, inner=5, shrink_steps=0, batch_size=128, trigger_norm=1000, alpha=0.9, gamma=1e-8, lr_decay=False)
    else: raise NotImplementedError()
    defense.detect()
elif args.defense == 'RNP':
    from other_defenses_tool_box.RNP import RNP
    if args.dataset == 'cifar10':
        defense = RNP(args, schedule=[10, 20], batch_size=128, momentum=0.9, weight_decay=5e-4, alpha=0.2, clean_threshold=0.20, unlearning_lr=0.01, recovering_lr=0.2, unlearning_epochs=20, recovering_epochs=20, pruning_by='number', pruning_max=0.90, pruning_step=0.01, max_CA_drop=0.5)
    else: raise NotImplementedError()
    defense.detect()
elif args.defense == "FeatureRE":
    from other_defenses_tool_box.feature_re import FeatureRE
    defense = FeatureRE(args)
    defense.detect()
elif args.defense == "CD":
    from other_defenses_tool_box.CD import CognitiveDistillation
    defense = CognitiveDistillation(args)
    defense.detect()
elif args.defense == "BaDExpert":
    from other_defenses_tool_box.bad_expert import BaDExpert
    defense = BaDExpert(args, defense_fpr=None)
    defense.detect()
else:
    raise NotImplementedError()

end_time = time.perf_counter()
print("Elapsed time: {:.2f}s".format(end_time - start_time))
