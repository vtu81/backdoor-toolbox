# Backdoor Toolbox

**A compact toolbox for backdoor attacks and defenses.**

## Features

> You may register your own attacks, defenses and visualization methods in the corresponding files and directories.

### Attacks

(This repository currently focuses on backdoor poisoning attacks.)

See [poison_tool_box/](poison_tool_box/) and [create_poisoned_set.py](create_poisoned_set.py).

**Clean**

- `none`: no attack

**Adaptive**

https://arxiv.org/abs/2205.13613
- `adaptive`: adaptive attack with a single general trigger
- `adaptive_mask`: adaptive blend attack with a single blending trigger
- `adaptive_k`: adaptive patch attack with `k` different triggers
- `adaptive_k_way`: adaptive attack with `k` pixel triggers
<!-- - `adaptive_blend`: adaptive attack with a single blending trigger -->

**Others**

- `basic`: basic attack with a general trigger (patch, blend, etc.)
- `badnet`: basic attack with badnet patch trigger, http://arxiv.org/abs/1708.06733
- `blend`: basic attack with a single blending trigger, https://arxiv.org/abs/1712.05526
- `dynamic`: http://arxiv.org/abs/2010.08138
- `clean_label`: http://arxiv.org/abs/1912.02771
- `SIG`: https://arxiv.org/abs/1902.11237
- `TaCT`: source specific attack, https://arxiv.org/abs/1908.00686
- `ISSBA`: invisible backdoor attack with sample-specific triggers, https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.pdf
- `refool`: reflection backdoor, http://arxiv.org/abs/2007.02343
- `WaNet`: imperceptible warping-based backdoor attack, http://arxiv.org/abs/2102.10369

### Defenses

**Poison Cleansers**

See [cleansers_tool_box/](cleansers_tool_box/) and [cleanser.py](cleanser.py).

- `CT`: confusion training, https://arxiv.org/abs/2205.13616
- `SCAn`: https://arxiv.org/abs/1908.00686
- `AC`: activation clustering, https://arxiv.org/abs/1811.03728
- `SS`: spectral signature, https://arxiv.org/abs/1811.00636
- `SPECTRE`: https://arxiv.org/abs/2104.11315
- `Strip` (modified as a poison cleanser): http://arxiv.org/abs/1902.06531

**Other Defenses**

See [other_defenses_tool_box/](other_defenses_tool_box/) and [other_defense.py](other_defense.py).

- `NC`: Neural Clenase, https://ieeexplore.ieee.org/document/8835365/
- `STRIP` (backdoor input filter): http://arxiv.org/abs/1902.06531
- `FP`: Fine-Pruning, http://arxiv.org/abs/1805.12185
- `ABL`: Anti-Backdoor Learning, https://arxiv.org/abs/2110.11571
- `NAD`: Neural Attention Distillation, https://arxiv.org/abs/2101.05930

### Visualization

Visualize the latent space of backdoor models. See [visualize.py](visualize.py).

- `tsne`: 2-dimensional T-SNE
- `pca`: 2-dimensional PCA
- `oracle`: fit the poison latent space with a SVM, see https://arxiv.org/abs/2205.13613

## Quick Start

Take launching and defending an Adaptive-Blend attack as an example:
```bash
# Create a clean set (for testing and some defenses)
python create_clean_set.py -dataset=cifar10

# Create a poisoned training set
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_mask -poison_rate=0.003 -cover_rate=0.003

# Train on the poisoned training set
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_mask -poison_rate=0.003 -cover_rate=0.003
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_mask -poison_rate=0.003 -cover_rate=0.003 -no_aug

# Visualize
## $METHOD = ['pca', 'tsne', 'oracle']
python visualize.py -method=$METHOD -dataset=cifar10 -poison_type=adaptive_mask -poison_rate=0.003 -cover_rate=0.003

# Cleanse poison train set with cleansers
## $CLEANSER = ['CT', 'SCAn', 'AC', 'SS', 'Strip', 'SPECTRE']
## Except for 'CT', you need to train poisoned backdoor models first.
python cleanser.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_mask -poison_rate=0.003 -cover_rate=0.003

# Retrain on cleansed set
## $CLEANSER = ['CT', 'SCAn', 'AC', 'SS', 'Strip', 'SPECTRE']
python train_on_cleansed_set.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_mask -poison_rate=0.003 -cover_rate=0.003

# Other defenses
## $DEFENSE = ['ABL', 'NC', 'STRIP', 'FP', 'NAD']
## Except for 'ABL', you need to train poisoned backdoor models first.
python other_defense.py -defense=$DEFENSE -dataset=cifar10 -poison_type=adaptive_mask -poison_rate=0.003 -cover_rate=0.003
```

**Notice**:
- `SPECTRE` is implemented in Julia. So you must install Julia and install dependencies before running SPECTRE, see [cleansers_tool_box/spectre/README.md](cleansers_tool_box/spectre/README.md) for configuration details.
- For `clean_label` attack, run [data/cifar10/clean_label/setup.sh](data/cifar10/clean_label/setup.sh) before the first time launching it.
- For `dynamic` attack, download pretrained generators `all2one_cifar10_ckpt.pth.tar` and `all2one_gtsrb_ckpt.pth.tar` to `[models/](models/) from https://github.com/VinAIResearch/input-aware-backdoor-attack-release before the first time launching it.

Some examples for creating other backdoor poison sets:
```bash
# No Poison
python create_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0
# Basic
python create_poisoned_set.py -dataset=cifar10 -poison_type=basic -trigger=firefox_corner_32.png -alpha=0.2 -poison_rate=0.003
# BadNet
python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.003
# Blend
python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.003
# Dynamic
python create_poisoned_set.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.003
# Clean Label
python create_poisoned_set.py -dataset=cifar10 -poison_type=clean_label -poison_rate=0.003
# SIG
python create_poisoned_set.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.003
# TaCT
python create_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.003 -cover_rate=0.003
# ISSBA
python create_poisoned_set.py -dataset=cifar10 -poison_type=ISSBA -poison_rate=0.003
# refool
python create_poisoned_set.py -dataset=cifar10 -poison_type=refool -poison_rate=0.003
# WaNet
python create_poisoned_set.py -dataset=cifar10 -poison_type=WaNet -poison_rate=0.003 -cover_rate=0.003
# Adaptive
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive -trigger=watermark_white_32.png -alpha=0.2 -poison_rate=0.003 -cover_rate=0.003
# Adaptive Blend
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_mask -poison_rate=0.003 -cover_rate=0.003
# Adaptive Patch
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_k -poison_rate=0.003 -cover_rate=0.006
# Adaptive K Way
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_k -poison_rate=0.003 -cover_rate=0.006
```

You can also:
- specify more details on the trigger selection
    - For `basic`, `blend`, `adaptive` and `adaptive_mask`:

        specify the opacity of the trigger by `-alpha=$ALPHA`.
    
    - For `basic`, `blend`, `clean_label`, `adaptive`, `adaptive_mask` and `TaCT`:
    
        specify the trigger by `-trigger=$TRIGGER_NAME`, where `$TRIGGER_NAME` is the file name of a 32x32 trigger mark image in [triggers/](triggers) (e.g., `-trigger=badnet_patch_32.png`).
    
    - For `basic`, `clean_label`, `adaptive` and `TaCT`:
    
        if another image named `mask_$TRIGGER_NAME` also exists in [triggers/](triggers), it will be used as the trigger mask. Otherwise, all black pixels of the trigger mark are not applied by default.

- train a vanilla model via
    ```bash
    python train_vanilla.py
    ```
- test a trained model via
    ```bash
    python test_model.py -dataset=cifar10 -poison_type=adaptive_mask -poison_rate=0.003 -cover_rate=0.003
    # other options include: -no_aug, -cleanser=$CLEANSER, -model_path=$MODEL_PATH, see our code for details
    ```
- enforce a fixed running seed via `-seed=$SEED` option
- change dataset to GTSRB via `-dataset=gtsrb` option
- change model architectures in [config.py](config.py)
- configure hyperparamters of other defenses in [other_defense.py](other_defense.py)
- see more configurations in [config.py](config.py)