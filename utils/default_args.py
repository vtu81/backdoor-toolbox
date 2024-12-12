parser_choices = {

    'dataset': ['gtsrb', 'cifar10', 'cifar100', 'imagenette', 'ember', 'imagenet'],
    'poison_type': [  # Poisoning attacks
        'basic', 'badnet', 'blend', 'dynamic', 'clean_label', 'TaCT', 'SIG', 'WaNet', 'refool', 'ISSBA',
        'adaptive_blend', 'adaptive_patch', 'adaptive_k_way', 'none', 'badnet_all_to_all', 'trojan', 'SleeperAgent',
        # Other attacks
        'trojannn', 'BadEncoder', 'SRA', 'bpp', 'WB'],
    # 'poison_rate': [0, 0.001, 0.002, 0.004, 0.005, 0.008, 0.01, 0.015, 0.02, 0.05, 0.1],
    # 'cover_rate': [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2],
    'poison_rate': [i / 1000.0 for i in range(0, 500)],
    'cover_rate': [i / 1000.0 for i in range(0, 500)],
    'cleanser': ['SCAn', 'AC', 'SS', 'Strip', 'CT', 'SPECTRE', 'SentiNet', 'Frequency'],
    'defense': ['BTIDBFU','ABL', 'NC', 'STRIP', 'FP', 'NAD', 'SentiNet', 'ScaleUp', 'SEAM', 'SFT', 'NONE', 'Frequency', 'AC', 'moth', 'IBAU', 'ANP', 'FeatureRE', 'AWM', 'RNP', 'CD', 'BaDExpert', 'IBD_PSC'],
}

parser_default = {
    'dataset': 'cifar10',
    'poison_type': 'badnet',
    'poison_rate': 0,
    'cover_rate': 0,
    'alpha': 0.2,
}

seed = 2333  # 999, 999, 666 (1234, 5555, 777)
