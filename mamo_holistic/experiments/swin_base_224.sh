#!/bin/bash

# This script is used to run the experiments to see resnet18, resnet34, resnet50 with 224x224 patches.

# Initialize Conda
if [ -d "$HOME/miniconda3" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
elif [ -d "$HOME/anaconda3" ]; then
    source $HOME/anaconda3/etc/profile.d/conda.sh
else
    echo "Conda not found in the default directories"
fi

# Activate the Mammo environment
conda activate Mammo

# Your commands here
echo "Mammo environment activated"


cd ..
echo "Current working directory: $(pwd)"

#run experiments for resnet18, resnet34, resnet50 with 224x224 patches

python train/train.py --config config_files/swin/swin_base_224.yaml