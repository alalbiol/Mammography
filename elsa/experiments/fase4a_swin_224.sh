#!/bin/bash
# =============================================================
# Fase IV-A | Swin Transformer | Búsqueda de Learning Rate
# Experimentos: E04_S1 (1e-4), E04_S2 (5e-5), E04_S3 (1e-5)
# Lanzar desde: experiments/
# Uso: nohup ./fase4a_swin.sh > logs_resnet/fase4a_swin.txt 2>&1 &
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

export WANDB_NOTES="Fase IV-A: Búsqueda LR en Swin Transformer. BS=125, SN, smoothed_CE, mixup=0.8. LRs: 1e-4 / 5e-5 / 1e-5."

# ------------------------------------------------------------------
# E04_S1 | Swin | LR = 1e-4
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E04_S1 | Swin | LR = 1e-4"
echo "================================================"
python train/train.py \
    --config_file config_files/swin/swin_224_LR1e-4.yaml \
    --job_name "E04_Swin_LR1e-4"

echo "E04_S1 finalizado."

# ------------------------------------------------------------------
# E04_S2 | Swin | LR = 5e-5
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E04_S2 | Swin | LR = 5e-5"
echo "================================================"
python train/train.py \
    --config_file config_files/swin/swin_224_LR5e-5.yaml \
    --job_name "E04_Swin_LR5e-5"

echo "E04_S2 finalizado."

# ------------------------------------------------------------------
# E04_S3 | Swin | LR = 1e-5
# ------------------------------------------------------------------
echo ""
echo "================================================"
echo " Lanzando E04_S3 | Swin | LR = 1e-5"
echo "================================================"
python train/train.py \
    --config_file config_files/swin/swin_224_LR1e-5.yaml \
    --job_name "E04_Swin_LR1e-5"

echo "E04_S3 finalizado."
echo ""
echo "=== Fase IV-A Swin completada ==="