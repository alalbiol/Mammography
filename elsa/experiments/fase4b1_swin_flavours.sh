#!/bin/bash
# =============================================================
# Fase IV-B.1 | Swin Transformer Flavours
# Experimentos: E05_S1 (Tiny), E05_S2 (Small), E05_S3 (Base)
# LR = 1e-4 (mejor de IV-A), BS=65 + accumulate_grad_batches=2
# Lanzar desde: experiments/
# Uso: nohup ./fase4b1_swin_flavours.sh > logs_swin/fase4b1_swin_flavours.txt 2>&1 &
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

export WANDB_NOTES="Fase IV-B.1: Comparativa flavours Swin Transformer. LR=1e-4, BS=65 x accumulate=2, 250 epochs."

# ------------------------------------------------------------------
# E05_S1 | Swin Tiny | LR = 1e-4
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E05_S1 | Swin Tiny | LR = 1e-4"
echo "================================================"
python train/train.py \
    --config_file config_files/swin/swin_tiny_224_LR1e-4.yaml \
    --job_name "E05_Swin_Tiny_LR1e-4"

echo "E05_S1 finalizado."

# ------------------------------------------------------------------
# E05_S2 | Swin Small | LR = 1e-4
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E05_S2 | Swin Small | LR = 1e-4"
echo "================================================"
python train/train.py \
    --config_file config_files/swin/swin_small_224_LR1e-4.yaml \
    --job_name "E05_Swin_Small_LR1e-4"

echo "E05_S2 finalizado."

# ------------------------------------------------------------------
# E05_S3 | Swin Base | LR = 1e-4
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E05_S3 | Swin Base | LR = 1e-4"
echo "================================================"
python train/train.py \
    --config_file config_files/swin/swin_base_224_LR1e-4.yaml \
    --job_name "E05_Swin_Base_LR1e-4"

echo "E05_S3 finalizado."
echo ""
echo "=== Fase IV-B.1 Swin Flavours completada ==="