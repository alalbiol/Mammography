!#/bin/bash

# This script is used to run the experiments for the Nikulin dataset with different sizes of patches.

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

# Run the experiments for the Nikulin dataset with different sizes of patches.
#enable Mammo conda environment
python train/train.py --config config_files/base_config.yaml --overrides config_files/nikulin/nikulin_224.yaml
python train/train.py --config config_files/base_config.yaml --overrides config_files/nikulin/nikulin_256.yaml
python train/train.py --config config_files/base_config.yaml --overrides config_files/nikulin/nikulin_512.yaml
