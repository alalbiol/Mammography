!#/bin/bash

# This script is used to run the experiments selecting only hard background patches for the loss computation.

# Initialize Conda
#check if the conda is installed either in miniconda3 or anaconda3
if [ -d "~/miniconda3" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -d "~/anaconda3" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
else
    echo "Conda not found in the default directories"
fi


# Activate the Mammo environment
conda activate Mammo

# Your commands here
echo "Mammo environment activated"


cd ..
echo "Current working directory: $(pwd)"

# Run the experiments for the Nikulin dataset with different sizes of patches.
#enable Mammo conda environment
python train/train.py --config config_files/base_config.yaml --overrides config_files/nikulin_topk/nikulin_224.yaml