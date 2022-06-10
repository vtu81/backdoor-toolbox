# Confusion Training

## Attacks

See [poison_tool_box/](poison_tool_box/).

**Adaptive**
- `adaptive`: adaptive attack with a single general trigger
- `adaptive_blend`: adaptive attack with a single blending trigger
- `adaptive_k`: adaptive attack with `k`=4 different triggers
- `adaptive_k_way`: adaptive attack with `k`=4 pixel triggers

**Others**
- `badnet`: basic attack with badnet patch trigger
- `blend`: basic attack with a single blending trigger
- `TaCT`: source specific attack

## Defenses

See [other_cleanses/](cleansers_tool_box/) and [other_defenses_tool_box/](other_defenses_tool_box/).

- `SCAn`
- `AC`: activation clustering
- `SS`: spectral signature
- `SPECTRE`
- `Strip` (Poison Cleanser)
- `Strip` (Input Filter)

## Visualization

See [visualize.py](visualize.py).

- `tsne`
- `pca`
- `oracle`

## Quick Start

To launch an Adaptive-Blend attack:
```bash
# Create a clean set
python create_clean_set.py -dataset=cifar10

# Create a poisoned training set
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005

# Train on the poisoned training set
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005 -no_aug

# Visualize
## $METHOD = ['pca', 'tsne', 'oracle']
python visualize.py -method=$METHOD -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005

# Defense
## $CLEANSER = ['SCAn', 'AC', 'SS', 'Strip', 'SPECTRE']
python other_cleanser.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005
## $DEFENSE = ['ABL', 'NC', 'STRIP', 'FP']
python other_defense.py -defense=$DEFENSE -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005

# Retrain on cleansed set
python train_on_cleansed_set.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.005 -cover_rate=0.005
```

**Notice**: SPECTRE is implemented in Julia. So you must install Julia and install dependencies before running SPECTRE, see [cleansers_tool_box/spectre/README.md](cleansers_tool_box/spectre/README.md) for configuration details.

Other poisoning attacks we compare in our papers:
```bash
# No Poison
python create_poisoned_set.py -dataset=cifar10 -poison_type=none -poison_rate=0
# TaCT
python create_poisoned_set.py -dataset=cifar10 -poison_type=TaCT -poison_rate=0.005 -cover_rate=0.005
# Blend
python create_poisoned_set.py -dataset=cifar10 -poison_type=blend -poison_rate=0.005
# K Triggers
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_k -poison_rate=0.005 -cover_rate=0
# Adaptive K
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_k -poison_rate=0.005 -cover_rate=0.01
```

You can also:
- specify details on the trigger (for `blend`, `adaptive`, `adaptive_blend`, `TaCT` attacks) via
    - `-alpha=$ALPHA`, the opacity of the trigger.
    - `-trigger=$TRIGGER_NAME`, where `$TRIGGER_NAME` is the name of a 32x32 trigger mark image in [triggers/](triggers). If another image named `mask_$TRIGGER_NAME` also exists in [triggers/](triggers), it will be used as the trigger mask. Otherwise by default, all black pixels of the trigger mark are not applied.
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
- change the target class, network architecture and other configurations in `config.py`