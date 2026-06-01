#!/bin/bash
# =============================================================
# Fase IV-B.2 | Swin Transformer V2 Large | LR = 1e-4
# Lanzar desde: experiments/
# Uso: nohup ./fase4b2_swinv2.sh > logs_swin/fase4b2_swinv2.txt 2>&1 &
# =============================================================

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

export WANDB_NOTES="Fase IV-B.2: Swin Transformer V2 Large. LR=1e-4, BS=65 x accumulate=2, 250 epochs, image_size=256."

echo ""
echo "================================================"
echo " Lanzando E06 | SwinV2 Large | LR = 1e-4"
echo "================================================"
python train/train.py \
    --config_file config_files/swin/swinv2_large_256_LR1e-4.yaml \
    --job_name "E06_SwinV2_Large_LR1e-4"

echo "E06 finalizado."
echo ""
echo "=== Fase IV-B.2 SwinV2 completada ==="