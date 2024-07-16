![assets/backdoor-toolbox.gif](assets/backdoor-toolbox.gif)

**Backdoor-Toolbox** is a compact toolbox that integrates various backdoor attacks and defenses. We designed our toolbox with a **shallow** function call stack, which makes it easy to **read** and **transplant** by other researchers. Most codes are adapted from the original attack/defense implementation. This repo is still under heavy updates. Welcome to make your contributions for attacks/defenses that have not yet been implemented!

## Features

> You may register your own attacks, defenses and visualization methods in the corresponding files and directories.

### Attacks

**Poisoning attacks**

See [poison_tool_box/](poison_tool_box/) and [create_poisoned_set.py](create_poisoned_set.py).

- `none`: no attack
- `basic`: basic attack with a general trigger (patch, blend, etc.)
- `badnet`: basic attack with badnet patch trigger, http://arxiv.org/abs/1708.06733
- `blend`: basic attack with a single blending trigger, https://arxiv.org/abs/1712.05526
- `trojan`: basic attack with the patch trigger from trojanNN, https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech
- `clean_label`: http://arxiv.org/abs/1912.02771
- `dynamic`: http://arxiv.org/abs/2010.08138
- `SIG`: https://arxiv.org/abs/1902.11237
- `ISSBA`: invisible backdoor attack with sample-specific triggers, https://openaccess.thecvf.com/content/ICCV2021/papers/Li_Invisible_Backdoor_Attack_With_Sample-Specific_Triggers_ICCV_2021_paper.pdf
- `WaNet`: imperceptible warping-based backdoor attack, http://arxiv.org/abs/2102.10369
- `refool`: reflection backdoor, http://arxiv.org/abs/2007.02343
- `TaCT`: source specific attack, https://arxiv.org/abs/1908.00686
- `adaptive_blend`: Adap-Blend attack with a single blending trigger, https://openreview.net/forum?id=_wSHsgrVali
- `adaptive_patch`: Adap-Patch attack with `k`=4 different patch triggers, https://openreview.net/forum?id=_wSHsgrVali
- `adaptive_k_way`: adaptive attack with `k` pixel triggers, https://openreview.net/forum?id=_wSHsgrVali
- `badnet_all_to_all`: All-to-all BadNet attack
- `SleeperAgent`: http://arxiv.org/abs/2106.08970
- ... (others to be incorporated)

**Other attacks**

See [other_attacks_tool_box/](other_attacks_tool_box/) and [other_attack.py](other_attack.py).

- `BadEncoder`: https://arxiv.org/abs/2108.00352
- `bpp`: BppAttack, https://arxiv.org/abs/2205.13383
- `SRA`: Subnet Replacement Attack, https://openaccess.thecvf.com/content/CVPR2022/papers/Qi_Towards_Practical_Deployment-Stage_Backdoor_Attack_on_Deep_Neural_Networks_CVPR_2022_paper.pdf
- `TrojanNN`: https://docs.lib.purdue.edu/cgi/viewcontent.cgi?article=2782&context=cstech
- `WB`: Wasserstein Backdoor, https://proceedings.neurips.cc/paper/2021/file/9d99197e2ebf03fc388d09f1e94af89b-Paper.pdf
- ... (others to be incorporated)


### Defenses

**Poison Cleansers**

See [cleansers_tool_box/](cleansers_tool_box/) and [cleanser.py](cleanser.py).

- `SCAn`: https://arxiv.org/abs/1908.00686
- `AC`: activation clustering, https://arxiv.org/abs/1811.03728
- `SS`: spectral signature, https://arxiv.org/abs/1811.00636
- `SPECTRE`: https://arxiv.org/abs/2104.11315
- `Strip` (modified as a poison cleanser): http://arxiv.org/abs/1902.06531
- `SentiNet` (modified as a poison cleanser): https://ieeexplore.ieee.org/abstract/document/9283822
- `Frequency`: https://openaccess.thecvf.com/content/ICCV2021/html/Zeng_Rethinking_the_Backdoor_Attacks_Triggers_A_Frequency_Perspective_ICCV_2021_paper.html
- `CT`: confusion training, https://arxiv.org/abs/2205.13616 (deprecated, refer to [the official repository](https://github.com/Unispac/Fight-Poison-With-Poison/tree/master) for the latest version)
- ... (others to be incorporated)

**Other Defenses**

See [other_defenses_tool_box/](other_defenses_tool_box/) and [other_defense.py](other_defense.py).

- `AC`: Activation Clustering, https://arxiv.org/abs/1811.03728
- `ANP`: Adversarial Neuron Pruning, https://arxiv.org/abs/2110.14430
- `ABL`: Anti-Backdoor Learning, https://arxiv.org/abs/2110.11571
- `AWM`: Adversarial Weight Masking, https://openreview.net/forum?id=Yb3dRKY170h
- `BaDExpert`: BaDExpert, https://openreview.net/forum?id=s56xikpD92
- `CD`: Cognitive Distillation, https://arxiv.org/pdf/2301.10908.pdf
- `FeatureRE`: FeatureRE, https://arxiv.org/abs/2210.15127
- `FP`: Fine-Pruning, http://arxiv.org/abs/1805.12185
- `Frequency`: Frequency, https://arxiv.org/abs/2104.03413
- `IBAU`: Implicit Backdoor Adversarial Unlearning, https://openreview.net/forum?id=MeeQkFYVbzW
- `MOTH`: Model Orthogonalization, https://ieeexplore.ieee.org/abstract/document/9833688?casa_token=omSZ9__sKPUAAAAA:Sl5_sFfDqRS1Ojih-mtswu0vi4utqc2eoLuzDSnZ7LK8WsaageSOXKC8h7Sg4DYiDAHt0VZIuPs
- `NAD`: Neural Attention Distillation, https://arxiv.org/abs/2101.05930
- `NC`: Neural Clenase, https://ieeexplore.ieee.org/document/8835365/
- `NONE`: NON-LinEarity, https://arxiv.org/abs/2202.06382
- `RNP`: Reconstructive Neuron Pruning, https://arxiv.org/pdf/2305.14876.pdf
- `ScaleUp`: SCALE-UP, https://arxiv.org/abs/2302.03251
- `SEAM`: SEAM, https://arxiv.org/abs/2212.04687
- `SentiNet`: https://ieeexplore.ieee.org/abstract/document/9283822
- `STRIP` (backdoor input filter): http://arxiv.org/abs/1902.06531
- `SFT`: Super-Fine-Tuning, https://arxiv.org/abs/2212.09067
- `IBD-PSC` (Input-level Backdoor Detection): https://arxiv.org/abs/2405.09786
- ... (others to be incorporated)

### Visualization

Visualize the latent space of backdoor models. See [visualize.py](visualize.py).

- `tsne`: 2-dimensional T-SNE
- `pca`: 2-dimensional PCA
- `oracle`: fit the poison latent space with a SVM, see https://arxiv.org/abs/2205.13613


## Dependency

This repository was developed with PyTorch 1.12.1, and should be compatible with PyTorch of newer versions. To set up the required environment, first manually install PyTorch with CUDA, and then install other packages via `pip install -r requirement.txt`.


## TODO before You Start

- Datasets:
  * Original CIFAR10 and GTSRB datasets would be automatically downloaded. 
  * ImageNet should be manually downloaded from [Kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) or other available sources. Then set up the local path to your ImageNet dataset via the `imagenet_dir` variable in [config.py](config.py).
- Before any experiments, first initialize the clean reserved data and validation data using command `python create_clean_set.py -dataset=$DATASET -clean_budget $N`, where `$DATASET = cifar10, gtsrb, ember, imagenet`, `$N = 2000` for `cifar10, gtsrb`, `$N = 5000` for `imagenet`.
- Before launching `clean_label` attack, run [data/cifar10/clean_label/setup.sh](data/cifar10/clean_label/setup.sh).
- Before launching `dynamic` attack, download pretrained generators `all2one_cifar10_ckpt.pth.tar` and `all2one_gtsrb_ckpt.pth.tar` to [models/](models/) from https://drive.google.com/file/d/1vG44QYPkJjlOvPs7GpCL2MU8iJfOi0ei/view?usp=sharing and https://drive.google.com/file/d/1x01TDPwvSyMlCMDFd8nG05bHeh1jlSyx/view?usp=sharing.
- `SPECTRE` baseline defense is implemented in Julia. To compare our defense with `SPECTRE`, you must install Julia and install dependencies before running SPECTRE, see [cleansers_tool_box/spectre/README.md](cleansers_tool_box/spectre/README.md) for configuration details.
- `Frequency` baseline defense is based on Tensorflow. If you would like to reproduce their results, please install Tensorflow (code is tested with Tensorflow 2.8.1 and should be compatible with newer versions) manually, after installing all the dependencies upon. We suggest you create and use a separate (conda) environment for it.


## Quick Start

For example, to launch and defend against the Adaptive-Blend attack:
```bash
# Create a poisoned training set
python create_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15

# Train on the poisoned training set
python train_on_poisoned_set.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

# Test the backdoor model
python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

# Visualize
## $METHOD = ['pca', 'tsne', 'oracle']
python visualize.py -method=$METHOD -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

# Cleanse with other cleansers
## Except for 'Frequency', you need to train poisoned backdoor models first.
## $CLEANSER = ['SCAn', 'AC', 'SS', 'Strip', 'SPECTRE', 'SentiNet', 'Frequency', etc.]
python cleanser.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

# Retrain on cleansed set
## $CLEANSER = ['SCAn', 'AC', 'SS', 'Strip', 'SPECTRE', 'SentiNet', etc.]
python train_on_cleansed_set.py -cleanser=$CLEANSER -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2

# Other defenses
## $DEFENSE = ['ABL', 'NC', 'NAD', 'STRIP', 'FP', 'SentiNet', 'IBD_PSC', etc.]
## Except for 'ABL', you need to train poisoned backdoor models first.
python other_defense.py -defense=$DEFENSE -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.003 -alpha 0.15 -test_alpha 0.2
```

<!-- **Notice**:
- `SPECTRE` is implemented in Julia. So you must install Julia and install dependencies before running SPECTRE, see [cleansers_tool_box/spectre/README.md](cleansers_tool_box/spectre/README.md) for configuration details.
- For `clean_label` attack, run [data/cifar10/clean_label/setup.sh](data/cifar10/clean_label/setup.sh) before the first time launching it.
- For `dynamic` attack, download pretrained generators `all2one_cifar10_ckpt.pth.tar` and `all2one_gtsrb_ckpt.pth.tar` to `[models/](models/) from https://drive.google.com/file/d/1vG44QYPkJjlOvPs7GpCL2MU8iJfOi0ei/view?usp=sharing and https://drive.google.com/file/d/1x01TDPwvSyMlCMDFd8nG05bHeh1jlSyx/view?usp=sharing before the first time launching it. -->
 <!-- https://github.com/VinAIResearch/input-aware-backdoor-attack-release before the first time launching it. -->

Some examples for creating other backdoor poison datasets:
```bash
# CIFAR10
python create_poisoned_set.py -dataset cifar10 -poison_type none
python create_poisoned_set.py -dataset cifar10 -poison_type badnet -poison_rate 0.003
python create_poisoned_set.py -dataset cifar10 -poison_type blend -poison_rate 0.003
python create_poisoned_set.py -dataset cifar10 -poison_type trojan -poison_rate 0.003
python create_poisoned_set.py -dataset cifar10 -poison_type clean_label -poison_rate 0.003
python create_poisoned_set.py -dataset cifar10 -poison_type SIG -poison_rate 0.02
python create_poisoned_set.py -dataset cifar10 -poison_type dynamic -poison_rate 0.003
python create_poisoned_set.py -dataset cifar10 -poison_type ISSBA -poison_rate 0.02
python create_poisoned_set.py -dataset cifar10 -poison_type WaNet -poison_rate 0.05 -cover_rate 0.1
python create_poisoned_set.py -dataset cifar10 -poison_type TaCT -poison_rate 0.003 -cover_rate 0.003
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_blend -poison_rate 0.003 -cover_rate 0.003 -alpha 0.15
python create_poisoned_set.py -dataset cifar10 -poison_type adaptive_patch -poison_rate 0.003 -cover_rate 0.006


# GTSRB
python create_poisoned_set.py -dataset gtsrb -poison_type none
python create_poisoned_set.py -dataset gtsrb -poison_type badnet -poison_rate 0.01
python create_poisoned_set.py -dataset gtsrb -poison_type blend -poison_rate 0.01
python create_poisoned_set.py -dataset gtsrb -poison_type trojan -poison_rate 0.01
python create_poisoned_set.py -dataset gtsrb -poison_type SIG -poison_rate 0.02
python create_poisoned_set.py -dataset gtsrb -poison_type dynamic -poison_rate 0.003
python create_poisoned_set.py -dataset gtsrb -poison_type WaNet -poison_rate 0.05 -cover_rate 0.1
python create_poisoned_set.py -dataset gtsrb -poison_type TaCT -poison_rate 0.005 -cover_rate 0.005
python create_poisoned_set.py -dataset gtsrb -poison_type adaptive_blend -poison_rate 0.003 -cover_rate 0.003 -alpha 0.15
python create_poisoned_set.py -dataset gtsrb -poison_type adaptive_patch -poison_rate 0.005 -cover_rate 0.01
```

### Additional Options and Configurations

You can also:
- specify more details on the trigger selection
    - For `basic`, `blend` and `adaptive_blend`:

        specify the opacity of the trigger by `-alpha=$ALPHA`.
    
    - For `basic`, `blend`, `clean_label`, `adaptive_blend` and `TaCT`:
    
        specify the trigger by `-trigger=$TRIGGER_NAME`, where `$TRIGGER_NAME` is the file name of a 32x32 trigger mark image in [triggers/](triggers) (e.g., `-trigger=badnet_patch_32.png`).
    
    - For `basic`, `clean_label` and `TaCT`:
    
        if another image named `mask_$TRIGGER_NAME` also exists in [triggers/](triggers), it will be used as the trigger mask. Otherwise, all black pixels of the trigger mark are not applied by default.

- test a trained model via
    ```bash
    python test_model.py -dataset=cifar10 -poison_type=adaptive_blend -poison_rate=0.003 -cover_rate=0.006 -alpha=0.15 -test_alpha=0.2
    # other options include: -no_aug, -cleanser=$CLEANSER, -model_path=$MODEL_PATH, see our code for details
    ```
- enforce a fixed running seed via `-seed=$SEED` option
- change dataset to GTSRB via `-dataset=gtsrb` option
- change model architectures in [config.py](config.py)
- configure hyperparamters of other defenses in [other_defense.py](other_defense.py)
- see more configurations in [config.py](config.py)


### Citation

If you find this toolbox useful for your research, please consider citing our work:

```
@inproceedings{qi2022revisiting,
  title={Revisiting the assumption of latent separability for backdoor defenses},
  author={Qi, Xiangyu and Xie, Tinghao and Li, Yiming and Mahloujifar, Saeed and Mittal, Prateek},
  booktitle={The eleventh international conference on learning representations},
  year={2022}
}

@inproceedings{qi2023towards,
  title={Towards a proactive $\{$ML$\}$ approach for detecting backdoor poison samples},
  author={Qi, Xiangyu and Xie, Tinghao and Wang, Jiachen T and Wu, Tong and Mahloujifar, Saeed and Mittal, Prateek},
  booktitle={32nd USENIX Security Symposium (USENIX Security 23)},
  pages={1685--1702},
  year={2023}
}

@article{xie2023badexpert,
  title={BaDExpert: Extracting Backdoor Functionality for Accurate Backdoor Input Detection},
  author={Xie, Tinghao and Qi, Xiangyu and He, Ping and Li, Yiming and Wang, Jiachen T and Mittal, Prateek},
  journal={arXiv preprint arXiv:2308.12439},
  year={2023}
}
```
