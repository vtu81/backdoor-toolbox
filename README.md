# Backdoor Toolbox

**A compact toolbox for backdoor attacks and defenses.**


### Attacks

(This repository currently focuses on backdoor poisoning attacks.)

See [poison_tool_box/](poison_tool_box/) and [create_poisoned_set.py](create_poisoned_set.py).

**Clean**
- `none`: no attack

**Adaptive**
- `adaptive`: adaptive attack with a single general trigger
- `adaptive_blend`: adaptive attack with a single blending trigger
- `adaptive_k`: adaptive attack with `k` different triggers
- `adaptive_k_way`: adaptive attack with `k` pixel triggers

**Others**
- `basic`: basic attack with a general trigger (patch, blend, etc.)
- `badnet`: basic attack with badnet patch trigger
- `blend`: basic attack with a single blending trigger
- `dynamic`
- `clean_label`
- `SIG`
- `TaCT`: source specific attack

### Defenses

**Poison Cleansers**

See [cleansers_tool_box/](cleansers_tool_box/) and [cleanser.py](cleanser.py).

- `CT`: confusion training
- `SCAn`
- `AC`: activation clustering
- `SS`: spectral signature
- `SPECTRE` 
- `Strip` (modified as a poison cleanser)

**Other Defenses**

See [other_defenses_tool_box/](other_defenses_tool_box/) and [other_defense.py](other_defense.py).

- `NC`: Neural Clenase
- `STRIP` (backdoor input filter)
- `FP`: FinePruning
- `ABL`: Anti-Backdoor Learning
- `NAD`: Neural Attention Distillation

### Visualization

Visualize the latent space of backdoor models. See [visualize.py](visualize.py).

- `tsne`: 2-dimensional T-SNE
- `pca`: 2-dimensional PCA
- `oracle`: fit the poison latent space with a SVM

## Quick Start

Take launching and defending an Adaptive-Blend attack as an example:
```bash
# Create a clean set (for testing and some defenses)
python create_clean_set.py -dataset=cifar10

# Create a poisoned training set
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005

# Train on the poisoned training set
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -no_aug

# Visualize
## $METHOD = ['pca', 'tsne', 'oracle']
python visualize.py -method=$METHOD -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005

# Cleanse poison train set with cleansers
## $CLEANSER = ['CT', 'SCAn', 'AC', 'SS', 'Strip', 'SPECTRE']
## Except for 'CT', you need to train poisoned backdoor models first.
python cleanser.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005

# Retrain on cleansed set
## $CLEANSER = ['CT', 'SCAn', 'AC', 'SS', 'Strip', 'SPECTRE']
python train_on_cleansed_set.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005

# Other defenses
## $DEFENSE = ['ABL', 'NC', 'STRIP', 'FP', 'NAD']
## Except for 'ABL', you need to train poisoned backdoor models first.
python other_defense.py -defense=$DEFENSE -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005
```

**Notice**:
- `SPECTRE` is implemented in Julia. So you must install Julia and install dependencies before running SPECTRE, see [cleansers_tool_box/spectre/README.md](cleansers_tool_box/spectre/README.md) for configuration details.
- For `clean_label` attack, run [data/cifar10/clean_label/setup.sh](data/cifar10/clean_label/setup.sh) before the first time launching it.
- For `dynamic` attack, download pretrained generators `all2one_cifar10_ckpt.pth.tar` and `all2one_gtsrb_ckpt.pth.tar` to `[models/](models/) from https://github.com/VinAIResearch/input-aware-backdoor-attack-release before the first time launching it.

Some examples for creating other backdoor poison set:
```bash
# No Poison
python create_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0
# Basic
python create_poisoned_set.py -dataset=cifar10 -poison_type=basic -trigger=firefox_corner_32.png -alpha=0.2 -poison_rate=0.01
# BadNet
python create_poisoned_set.py -dataset=cifar10 -poison_type=badnet -poison_rate=0.01
# Blend
python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.01
# Dynamic
python create_poisoned_set.py -dataset=cifar10 -poison_type=dynamic -poison_rate=0.01
# Clean Label
python create_poisoned_set.py -dataset=cifar10 -poison_type=clean_label -poison_rate=0.005
# SIG
python create_poisoned_set.py -dataset=cifar10 -poison_type=SIG -poison_rate=0.02
# TaCT
python create_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.02 -cover_rate=0.01
# Adaptive
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive -trigger=watermark_white_32.png -alpha=0.2 -poison_rate=0.005 -cover_rate=0.005
# Adaptive Blend
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005
# Adaptive K
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_k -poison_rate=0.005 -cover_rate=0.01
# Adaptive K Way
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_k -poison_rate=0.005 -cover_rate=0.01
```

You can also:
- specify more details on the trigger selection
    - For `basic`, `blend`, `adaptive` and `adaptive_blend`:

        specify the opacity of the trigger by `-alpha=$ALPHA`.
    
    - For `basic`, `blend`, `clean_label`, `adaptive`, `adaptive_blend` and `TaCT`:
    
        specify the trigger by `-trigger=$TRIGGER_NAME`, where `$TRIGGER_NAME` is the file name of a 32x32 trigger mark image in [triggers/](triggers) (e.g., `-trigger=badnet_patch_32.png`).
    
    - For `basic`, `clean_label`, `adaptive` and `TaCT`:
    
        if another image named `mask_$TRIGGER_NAME` also exists in [triggers/](triggers), it will be used as the trigger mask. Otherwise, all black pixels of the trigger mark are not applied by default.

- train a vanilla model via
    ```bash
    python train_vanilla.py
    ```
- test a trained model via
    ```bash
    python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005
    # other options include: -no_aug, -cleanser=$CLEANSER, -model_path=$MODEL_PATH, see our code for details
    ```
- enforce a fixed running seed via `-seed=$SEED` option
- change dataset to GTSRB via `-dataset=gtsrb` option
- configure hyperparamters of other defenses in [other_defense.py](other_defense.py)
- see more configurations in `config.py`