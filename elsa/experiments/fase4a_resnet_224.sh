#!/bin/bash
# =============================================================
# Fase IV-A | ResNet50 | Búsqueda de Learning Rate
# Experimentos: E04_R1 (1e-3), E04_R2 (1e-4), E04_R3 (5e-5)
# Lanzar desde: experiments/
# Uso: nohup ./fase4a_resnet.sh > logs_resnet/fase4a_resnet.txt 2>&1 &
# =============================================================

# Inicializar Conda
if [ -d "$HOME/miniconda3" ]; then
    source $HOME/miniconda3/etc/profile.d/conda.sh
elif [ -d "$HOME/anaconda3" ]; then
    source $HOME/anaconda3/etc/profile.d/conda.sh
else
    echo "Conda not found"; exit 1
fi

conda activate proymam
echo "proymam environment activated"

cd ..
echo "Working directory: $(pwd)"

# Notas comunes para WandB
export WANDB_NOTES="Fase IV-A: Búsqueda LR en ResNet50. BS=125, SN, CE. LRs: 1e-3 / 1e-4 / 5e-5."

# ------------------------------------------------------------------
# E04_R1 | ResNet50 | LR = 1e-3
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E04_R1 | ResNet50 | LR = 1e-3"
echo "================================================"
NUM_EPOCHS=250 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/resnet/resnet_50_LR1e3.yaml \
    --batch_size 125 \
    --job_name "E04_Resnet_LR1e-3"

echo "E04_R1 finalizado."

# ------------------------------------------------------------------
# E04_R2 | ResNet50 | LR = 1e-4
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E04_R2 | ResNet50 | LR = 1e-4"
echo "================================================"
NUM_EPOCHS=250 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/resnet/resnet_50_LR1e4.yaml \
    --batch_size 125 \
    --job_name "E04_Resnet_LR1e-4"

echo "E04_R2 finalizado."

# ------------------------------------------------------------------
# E04_R3 | ResNet50 | LR = 5e-5
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E04_R3 | ResNet50 | LR = 5e-5"
echo "================================================"
NUM_EPOCHS=250 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/resnet/resnet_50_LR5e5.yaml \
    --batch_size 125 \
    --job_name "E04_Resnet_LR5e-5"

echo "E04_R3 finalizado."
echo ""
echo "=== Fase IV-A ResNet completada ==="