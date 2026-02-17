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

# Your commands here
echo "proymam environment activated"


cd ..
echo "Current working directory: $(pwd)"


export max_epochs=1,
export limit_train_batches=10,  # Only 10 training batcheslimit_val_batches=2,     # Only 2 validation batches
export limit_test_batches=1     # Only 1 test batch


for BATCH_SIZE in 10 40 50
for MODEL in resnet_18 resnet_34 resnet_50
do
    do
        export BATCH_SIZE
        export MODEL

         NUM_EPOCHS=$(( 35 * 160 / BATCH_SIZE ))
        export NUM_EPOCHS
        
        echo "Running with MODEL=$MODEL Running with BATCH_SIZE=$BATCH_SIZE"
        export JOB_NAME="${MODEL}_BS_${BATCH_SIZE}"
        python train/train.py --config config_files/base_config.yaml
    done
done

