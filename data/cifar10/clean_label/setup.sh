#!/bin/bash

wget https://github.com/MadryLab/label-consistent-backdoor-code/releases/download/v1.0/clean_dataset.tar.bz2
tar -vxjf clean_dataset.tar.bz2
rm clean_dataset.tar.bz2

wget https://github.com/MadryLab/label-consistent-backdoor-code/releases/download/v1.0/fully_poisoned_training_datasets.tar.bz2.aa
wget https://github.com/MadryLab/label-consistent-backdoor-code/releases/download/v1.0/fully_poisoned_training_datasets.tar.bz2.ab
wget https://github.com/MadryLab/label-consistent-backdoor-code/releases/download/v1.0/fully_poisoned_training_datasets.tar.bz2.ac
cat fully_poisoned_training_datasets.tar.bz2.* > fully_poisoned_training_datasets.tar.bz2
rm fully_poisoned_training_datasets.tar.bz2.*
tar -vxjf fully_poisoned_training_datasets.tar.bz2
rm fully_poisoned_training_datasets.tar.bz2

