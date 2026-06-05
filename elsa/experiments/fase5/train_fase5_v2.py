"""
experiments/fase5/train_fase5_v2.py
=====================================
E05_v2 — Aumentado masivo para combatir el overfitting de E05_v1.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "train"))

from experiments.fase5.swin_image_model import SwinBreastCancerLarge
import models.model_selector as ms

_original_get_image_model = ms.get_image_model

def _patched_get_image_model(model_name, num_classes=2, **kwargs):
    if model_name == "SwinBreastCancerLarge":
        print("Using SwinBreastCancerLarge (Fase V)")
        return SwinBreastCancerLarge(num_classes=num_classes, **kwargs)
    return _original_get_image_model(model_name, num_classes=num_classes, **kwargs)

ms.get_image_model = _patched_get_image_model

from data.ddsm_dataset import DDSMImageDataModule as _OriginalDataModule
from experiments.fase5.augmentations_v2 import (
    get_geometric_transform_v2,
    get_intensity_transform_v2,
)
from data.ddsm_dataset import DDSM_Image_Dataset, BalancedBatchSampler
from torch.utils.data import DataLoader
from utils.load_config import get_parameter


class DDSMImageDataModuleV2(_OriginalDataModule):
    def train_dataloader(self):
        geometric_transform = get_geometric_transform_v2()
        intensity_transform = get_intensity_transform_v2()

        dataset = DDSM_Image_Dataset(
            self.train_csv,
            self.ddsm_annotations,
            self.ddsm_root,
            convert_to_rgb=False,
            subset_size=self.subset_size_train,
            random_seed=self.random_seed,
            geometrical_transform=geometric_transform,
            intensity_transform=intensity_transform,
            return_mask=self.return_mask,
            use_all_images=True,
        )

        if self.balanced_patches:
            print("Using BalancedBatchSampler")
            sampler = BalancedBatchSampler(dataset, batch_size=self.batch_size)
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
        )


from train.train_image import DDSMImageClassifier, get_logger, create_callbacks
from utils.load_config import load_config, get_parameter
from utils.utils import str_to_bool

import torch
import pytorch_lightning as pl
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--overrides",   default=None)
    parser.add_argument("--logger",      type=str_to_bool, default=True)
    parser.add_argument("--task",        type=str, default="train",
                        choices=["train", "validate", "test"])
    args = parser.parse_args()

    config = load_config(args.config_file, override_file=args.overrides)

    GPU_TYPE = get_parameter(config, ["General", "gpu_type"], default="None")
    if GPU_TYPE in ("RTX 3090", "RTX 5090"):
        torch.set_float32_matmul_precision("medium")

    model       = DDSMImageClassifier(config=config)
    data_module = DDSMImageDataModuleV2(config=config)
    logger      = get_logger(config) if args.logger else None
    callbacks   = create_callbacks(config)

    trainer_kwargs = get_parameter(config, ["Trainer"], mode="default", default={})
    trainer = pl.Trainer(
        logger      = logger,
        callbacks   = callbacks,
        accelerator = "gpu" if torch.cuda.is_available() else "cpu",
        **trainer_kwargs,
    )

    if args.task == "train":
        trainer.fit(model, data_module)
    elif args.task == "validate":
        trainer.validate(model, data_module)
    elif args.task == "test":
        trainer.test(model, data_module)