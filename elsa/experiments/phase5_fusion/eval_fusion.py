"""
experiments/phase5_fusion/eval_fusion.py
========================================
Fase V – Resolución y fusión.

Carga el mejor checkpoint de la Fase IV (Swin Transformer),
infiere sobre mamografías completas mediante sliding window y
compara varias estrategias de fusión de parches para obtener
una puntuación a nivel de imagen.

Estrategias evaluadas:
    max        → score máximo entre todos los parches
    mean       → media de todos los parches
    top_k_mean → media de los k parches más sospechosos
    lse        → Log-Sum-Exp (aproximación suave al máximo)

Uso:
    python eval_fusion.py \
        --ckpt  /home/eblanov/tmp/logs/E04_Swin_LR1e-4/best_epoch=XX-val_auroc=0.XXXX.ckpt \
        --config experiments/phase5_fusion/config_phase5.yaml \
        --threshold 0.90
"""

import argparse
import pathlib
import sys

# ── path del proyecto ────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "train"))

# ── imports del proyecto ─────────────────────────────────────────────────────
from utils.load_config import load_config, get_parameter
from train.train import DDSMPatchClassifier
from data.sliding_window_dataset import SlidingWindowImageDataset, sliding_window_collate_fn

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import wandb


# ─────────────────────────────────────────────────────────────────────────────
# Estrategias de fusión
# ─────────────────────────────────────────────────────────────────────────────

def fuse_max(cancer_probs: np.ndarray) -> float:
    """Score máximo entre todos los parches."""
    return float(cancer_probs.max())


def fuse_mean(cancer_probs: np.ndarray) -> float:
    """Media de todos los parches."""
    return float(cancer_probs.mean())


def fuse_top_k_mean(cancer_probs: np.ndarray, k: int) -> float:
    """Media de los k parches con mayor probabilidad de cáncer."""
    k = min(k, len(cancer_probs))
    top_k = np.partition(cancer_probs, -k)[-k:]
    return float(top_k.mean())


def fuse_lse(cancer_probs: np.ndarray, temperature: float = 1.0) -> float:
    """
    Log-Sum-Exp: aproximación suave al máximo.
        LSE(p) = (1/T) * log( (1/N) * sum( exp(T * p_i) ) )
    Con T grande → se aproxima al máximo.
    Con T=1      → media geométrica ponderada.
    """
    scores = cancer_probs / temperature
    # resta el máximo para estabilidad numérica
    scores_shifted = scores - scores.max()
    lse = scores.max() + np.log(np.mean(np.exp(scores_shifted)))
    return float(lse)


FUSION_STRATEGIES = {
    "max":          lambda p: fuse_max(p),
    "mean":         lambda p: fuse_mean(p),
    "top_5_mean":   lambda p: fuse_top_k_mean(p, k=5),
    "top_10_mean":  lambda p: fuse_top_k_mean(p, k=10),
    "top_20_mean":  lambda p: fuse_top_k_mean(p, k=20),
    "lse_T1":       lambda p: fuse_lse(p, temperature=1.0),
    "lse_T5":       lambda p: fuse_lse(p, temperature=5.0),
    "lse_T10":      lambda p: fuse_lse(p, temperature=10.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluación en punto de operación  (réplica de eval_operating_point.py)
# ─────────────────────────────────────────────────────────────────────────────

def operating_point_metrics(labels: np.ndarray,
                             scores: np.ndarray,
                             sens_target: float = 0.90):
    """
    Devuelve (auroc, sensitivity, specificity, threshold) en el punto
    donde sensitivity >= sens_target.
    """
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    idx = np.searchsorted(tpr, sens_target)
    if idx >= len(thresholds):
        idx = len(thresholds) - 1

    sens = float(tpr[idx])
    spec = float(1 - fpr[idx])
    thr  = float(thresholds[idx])
    return auroc, sens, spec, thr


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia: obtiene probabilidades de cáncer por parche para cada imagen
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_all_images(model, loader, device, mini_batch_size: int = 64):
    """
    Itera sobre el DataLoader (batch_size=1, una imagen a la vez) y
    devuelve dos listas paralelas:
        image_cancer_probs : list[ np.ndarray(N_parches,) ]  prob. cáncer por parche
        image_labels       : list[int]                       etiqueta binaria de la imagen
        image_ids          : list[str]
    """
    image_cancer_probs = []
    image_labels       = []
    image_ids          = []

    model.eval()

    for patches, label, image_id in loader:
        # patches: (N_parches, C, P, P)
        patches = patches.to(device)

        # Inferencia en mini-lotes para no saturar la GPU
        all_probs = []
        for i in range(0, len(patches), mini_batch_size):
            mb = patches[i : i + mini_batch_size]
            logits = model(mb)                              # (mb, num_classes)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

        probs_all = np.concatenate(all_probs, axis=0)       # (N_parches, num_classes)

        # Probabilidad de cáncer = P(MASS_MAL) + P(CALC_MAL)  (clases 3 y 4)
        cancer_prob = probs_all[:, 3] + probs_all[:, 4]    # (N_parches,)

        image_cancer_probs.append(cancer_prob)
        image_labels.append(int(label))
        image_ids.append(image_id)

    return image_cancer_probs, image_labels, image_ids


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(ckpt_path: str, config_path: str, sens_target: float = 0.90):

    config = load_config(config_path)

    # ── WandB (opcional) ────────────────────────────────────────────────────
    use_wandb = get_parameter(config, ["Logger", "type"], default=None) == "wandb"
    if use_wandb:
        run = wandb.init(
            project = get_parameter(config, ["Logger", "project"]),
            name    = get_parameter(config, ["Logger", "name"]),
            config  = {
                "ckpt":       ckpt_path,
                "sens_target": sens_target,
                "patch_size":  get_parameter(config, ["Datamodule", "patch_size"]),
                "stride":      get_parameter(config, ["Datamodule", "stride"]),
            }
        )

    # ── Modelo ──────────────────────────────────────────────────────────────
    patch_config_path = source_root / get_parameter(config, ["Phase4", "patch_config"])
    patch_config      = load_config(str(patch_config_path))

    model  = DDSMPatchClassifier.load_from_checkpoint(ckpt_path, config=patch_config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    print(f"Modelo cargado desde {ckpt_path}  →  device: {device}")

    # ── Dataset + DataLoader ─────────────────────────────────────────────────
    source_root = pathlib.Path(get_parameter(config, ["General", "source_root"], default=""))

    val_csv   = source_root / get_parameter(config, ["Datamodule", "val_csv"])
    annot_gz  = source_root / get_parameter(config, ["Datamodule", "ddsm_annotations"])
    root_dir  = get_parameter(config, ["Datamodule", "ddsm_root"])
    patch_size = get_parameter(config, ["Datamodule", "patch_size"])
    stride     = get_parameter(config, ["Datamodule", "stride"])
    convert_to_rgb  = get_parameter(config, ["Datamodule", "convert_to_rgb"], default=True)
    normalize_input = get_parameter(config, ["Datamodule", "normalize_input"], default=True)
    subset_size     = get_parameter(config, ["Datamodule", "subset_size"],     default=None)
    mini_batch_size = get_parameter(config, ["Datamodule", "mini_batch_size"], default=64)

    dataset = SlidingWindowImageDataset(
        val_csv         = str(val_csv),
        annot_gz        = str(annot_gz),
        root_dir        = root_dir,
        patch_size      = patch_size,
        stride          = stride,
        convert_to_rgb  = convert_to_rgb,
        normalize_input = normalize_input,
        subset_size     = subset_size,
    )

    loader = DataLoader(
        dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = get_parameter(config, ["Datamodule", "num_workers"], default=4),
        collate_fn  = sliding_window_collate_fn,
    )

    # ── Inferencia ───────────────────────────────────────────────────────────
    print(f"\nInferiendo sobre {len(dataset)} imágenes completas…")
    image_cancer_probs, image_labels, image_ids = infer_all_images(
        model, loader, device, mini_batch_size=mini_batch_size
    )

    labels_arr = np.array(image_labels)   # (M,)
    print(f"  → {labels_arr.sum()} imágenes malignas / {(labels_arr == 0).sum()} normales")

    # ── Comparativa de estrategias de fusión ─────────────────────────────────
    header = f"\n{'Estrategia':<20} {'AUROC':>7} {'Sens@{:.0f}%'.format(sens_target*100):>10} {'Spec':>7} {'Thr':>8}"
    print(header)
    print("─" * 58)

    wandb_results = {}
    best_auroc    = -1
    best_strategy = None

    for strategy_name, fusion_fn in FUSION_STRATEGIES.items():
        image_scores = np.array([fusion_fn(cp) for cp in image_cancer_probs])

        auroc, sens, spec, thr = operating_point_metrics(
            labels_arr, image_scores, sens_target
        )

        print(f"{strategy_name:<20} {auroc:>7.4f} {sens:>10.4f} {spec:>7.4f} {thr:>8.4f}")

        wandb_results[f"{strategy_name}/auroc"] = auroc
        wandb_results[f"{strategy_name}/sens"]  = sens
        wandb_results[f"{strategy_name}/spec"]  = spec

        if auroc > best_auroc:
            best_auroc    = auroc
            best_strategy = strategy_name

    print("─" * 58)
    print(f"Mejor estrategia: {best_strategy}  →  AUROC = {best_auroc:.4f}")

    if use_wandb:
        wandb.log(wandb_results)
        wandb.summary["best_strategy"] = best_strategy
        wandb.summary["best_auroc"]    = best_auroc
        wandb.finish()

    return wandb_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fase V – Evaluación de estrategias de fusión a nivel de imagen"
    )
    parser.add_argument("--ckpt",      required=True,  help="Ruta al checkpoint .ckpt de Fase IV")
    parser.add_argument("--config",    required=True,  help="Ruta al config_phase5.yaml")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Sensibilidad objetivo (default: 0.90)")
    args = parser.parse_args()

    evaluate(args.ckpt, args.config, args.threshold)