#!/bin/bash
# =============================================================
# Fase IV-A | Nikulin | Búsqueda de Learning Rate
# Experimentos: E04_N1 (1e-3), E04_N2 (1e-4), E04_N3 (5e-5)
# Lanzar desde: experiments/
# Uso: nohup ./fase4a_nikulin.sh > logs_resnet/fase4a_nikulin.txt 2>&1 &
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

export WANDB_NOTES="Fase IV-A: Búsqueda LR en Nikulin. BS=125, SN, CE. LRs: 1e-3 / 1e-4 / 5e-5."

# ------------------------------------------------------------------
# E04_N1 | Nikulin | LR = 1e-3
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E04_N1 | Nikulin | LR = 1e-3"
echo "================================================"
NUM_EPOCHS=250 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/nikulin/nikulin_224_LR1e-3.yaml \
    --batch_size 125 \
    --job_name "E04_N1_Nikulin_LR1e-3"

echo "E04_N1 finalizado."

# ------------------------------------------------------------------
# E04_N2 | Nikulin | LR = 1e-4
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E04_N2 | Nikulin | LR = 1e-4"
echo "================================================"
NUM_EPOCHS=250 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/nikulin/nikulin_224_LR1e-4.yaml \
    --batch_size 125 \
    --job_name "E04_N2_Nikulin_LR1e-4"

echo "E04_N2 finalizado."

# ------------------------------------------------------------------
# E04_N3 | Nikulin | LR = 5e-5
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E04_N3 | Nikulin | LR = 5e-5"
echo "================================================"
NUM_EPOCHS=250 python train/train.py \
    --config_file config_files/base_config.yaml \
    --overrides config_files/nikulin/nikulin_224_LR5e-5.yaml \
    --batch_size 125 \
    --job_name "E04_N3_Nikulin_LR5e-5"

echo "E04_N3 finalizado."
echo ""
echo "=== Fase IV-A Nikulin completada ==="