"""
Evalúa un checkpoint y reporta métricas en el punto de operación
donde la sensibilidad macro >= umbral_objetivo (por defecto 90%).

Uso:
    python eval_operating_point.py \
        --ckpt /home/eblanov/tmp/logs/model_resnet50_224/best_model_epoch_epoch=46_loss_val_loss=0.481_auroc_val_auroc=0.955.ckpt \
        --config resources/config.yaml \
        --threshold 0.90
"""
import pathlib, sys

path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(path))
sys.path.append(str(path / "train"))

# A partir de aquí los imports del proyecto
from utils.load_config import load_config
from data.ddsm_dataset import DDSMPatchDataModule
from train.train import DDSMPatchClassifier

import argparse
import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import pathlib, sys

# Añade el path del proyecto
path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(path))


CLASS_NAMES = ['NORMAL', 'MASS_BENIGN', 'CALC_BENIGN', 'MASS_MALIGNANT', 'CALC_MALIGNANT']

def evaluate(ckpt_path, config_path, sens_target=0.90):
    config = load_config(config_path)
    model = DDSMPatchClassifier.load_from_checkpoint(ckpt_path, config=config)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    dm = DDSMPatchDataModule(config=config)
    dm.setup()
    loader = dm.val_dataloader()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(device), batch[1]
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())

    probs  = np.concatenate(all_probs)   # (N, 5)
    labels = np.concatenate(all_labels)  # (N,)

    print(f"\n{'Clase':<25} {'AUROC':>7} {'Sens@target':>12} {'Spec@target':>12} {'Threshold':>10}")
    print("-" * 70)

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        binary_labels = (labels == i).astype(int)
        if binary_labels.sum() == 0:
            continue
        score = probs[:, i]
        auroc = roc_auc_score(binary_labels, score)
        fpr, tpr, thresholds = roc_curve(binary_labels, score)

        # Encuentra el umbral con sensibilidad >= sens_target
        idx = np.searchsorted(tpr, sens_target)
        if idx >= len(thresholds):
            idx = len(thresholds) - 1
        sens = tpr[idx]
        spec = 1 - fpr[idx]
        thr  = thresholds[idx]
        results[name] = dict(auroc=auroc, sens=sens, spec=spec, threshold=thr)
        print(f"{name:<25} {auroc:>7.4f} {sens:>12.4f} {spec:>12.4f} {thr:>10.4f}")

    # Métricas macro (excluyendo NORMAL si quieres)
    lesion_classes = [k for k in results if k != 'NORMAL']
    macro_auroc = np.mean([results[k]['auroc'] for k in lesion_classes])
    macro_sens  = np.mean([results[k]['sens']  for k in lesion_classes])
    macro_spec  = np.mean([results[k]['spec']  for k in lesion_classes])
    print("-" * 70)
    print(f"{'MACRO (sin NORMAL)':<25} {macro_auroc:>7.4f} {macro_sens:>12.4f} {macro_spec:>12.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",      required=True)
    parser.add_argument("--config",    required=True)
    parser.add_argument("--threshold", type=float, default=0.90)
    args = parser.parse_args()
    evaluate(args.ckpt, args.config, args.threshold)