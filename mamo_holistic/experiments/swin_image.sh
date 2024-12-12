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

python train/train_image.py --config config_files/swin/swin_224.yaml    \
            --overrides  config_files/swin/swin_variants/swin_base_patch4_window7_224.yaml

python train/train_image.py --config config_files/swin/swin_224.yaml    \
            --overrides  config_files/swin/swin_variants/swin_s3_base_224.yaml

python train/train_image.py --config config_files/swin/swin_224.yaml    \
            --overrides  config_files/swin/swin_variants/swin_s2_small_224.yaml

python train/train_image.py --config config_files/swin/swin_224.yaml    \
            --overrides  config_files/swin/swin_variants/swin_s2_tiny_224.yaml

python train/train_image.py --config config_files/swin/swin_224.yaml    \
            --overrides  config_files/swin/swin_variants/swin_tiny_patch4_window7_224.yaml

python train/train_image.py --config config_files/swin/swin_224.yaml    \
            --overrides  config_files/swin/swin_variants/swin_cr_small_224.yaml

python train/train_image.py --config config_files/swin/swin_224.yaml    \
            --overrides  config_files/swin/swin_variants/swin_cr_small_ns_224

python train/train_image.py --config config_files/swin/swin_224.yaml    \
            --overrides  config_files/swin/swin_variants/swin_cr_tiny_224.yaml

python train/train_image.py --config config_files/swin/swin_224.yaml    \
            --overrides  config_files/swin/swin_variants/swin_cr_tiny_ns_224.yaml


