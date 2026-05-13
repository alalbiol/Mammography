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

# Activate the proymam environment
conda activate proymam

# Your commands here
echo "proymam environment activated"


cd ..
echo "Current working directory: $(pwd)"

# 1. ESCRIBE AQUÍ TU COMENTARIO PARA WANDB
export WANDB_NOTES="Exp E03: ¿Focal Loss vs Cross-Entropy? Probando FL con gamma=0.5 y alpha. ResNet50, parches 224x224, Num_Epochs=250, SN."

# 2. EJECUTA EL EXPERIMENTO

# alpha=[1,1,1,1,1]
NUM_EPOCHS=250 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/resnet/resnet_50_FL_1.yaml \
    --batch_size 125 \
    --job_name "E03_FLG05_A11111" 

# alpha=[0.5,1,1,1,1]
NUM_EPOCHS=250 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/resnet/resnet_50_FL_051.yaml \
    --batch_size 125 \
    --job_name "E03_FLG05_A051111" 

# alpha=[0.5,1,1,2,2]
NUM_EPOCHS=250 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/resnet/resnet_50_FL_052.yaml \
    --batch_size 125 \
    --job_name "E03_FLG05_A051122" 


# alpha=[0.25,1,1,1,1]
NUM_EPOCHS=250 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/resnet/resnet_50_FL_025.yaml \
    --batch_size 125 \
    --job_name "E03_FLG05_A0251111" 


# alpha=[1,1,1,2,2]
NUM_EPOCHS=250 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/resnet/resnet_50_FL_12.yaml \
    --batch_size 125 \
    --job_name "E03_FLG05_A11122" 

