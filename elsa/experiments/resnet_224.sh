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
conda activate proymam
echo "proymam environment activated"


cd ..
echo "Current working directory: $(pwd)"

# 1. ESCRIBE AQUÍ TU COMENTARIO PARA WANDB
export WANDB_NOTES="Exp E01: ¿Batch optimo? Probando BS=75. ResNet50, parches 224x224, Num_Epochs=50."

# 2. LANZA EL EXPERIMENTO
NUM_EPOCHS=50 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/resnet/resnet_50.yaml \
    --batch_size 75 \
    --job_name "E01_BS_75"

#run experiments for resnet18, resnet34, resnet50 with 224x224 patches
#enable Mammo conda environment
#python train/train.py --config config_files/base_config.yaml --overrides config_files/resnet/resnet_34.yaml
# python train/train.py --config config_files/base_config.yaml --overrides config_files/resnet/resnet_50.yaml
