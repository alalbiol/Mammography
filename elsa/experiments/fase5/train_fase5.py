import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "train"))

# El patch DEBE ir antes de importar train_image para que funcione
from experiments.fase5.swin_image_model import SwinBreastCancerLarge
import models.model_selector as ms

_original_get_image_model = ms.get_image_model

def _patched_get_image_model(model_name, num_classes=2, **kwargs):
    if model_name == "SwinBreastCancerLarge":
        print("Using SwinBreastCancerLarge (Fase V)")
        return SwinBreastCancerLarge(num_classes=num_classes, **kwargs)
    return _original_get_image_model(model_name, num_classes=num_classes, **kwargs)

ms.get_image_model = _patched_get_image_model

from train.train_image import (
    DDSMImageClassifier,
    DDSMImageDataModule,
    get_logger,
    create_callbacks,
)

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
    data_module = DDSMImageDataModule(config=config)
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
