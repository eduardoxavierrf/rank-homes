"""Fine-tune RF-DETR Small on the converted ADE20K Rooms dataset."""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def training_settings(args):
    train_annotations = args.dataset / "train" / "_annotations.coco.json"
    categories = json.loads(train_annotations.read_bytes())["categories"]
    class_names = [item["name"] for item in sorted(categories, key=lambda item: item["id"])]
    return {
        "dataset_dir": str(args.dataset.resolve()),
        "output_dir": str(args.output.resolve()),
        "dataset_file": "roboflow",
        "class_names": class_names,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.accumulate,
        "num_workers": args.workers,
        "lr_scheduler": "cosine",
        "seed": 42,
        "eval_max_dets": 100,
        "amp_dtype": "bf16",
        "run_test": False,
        "tensorboard": False,
    }


def run_smoke_test(settings, model_settings):
    from rfdetr.config import RFDETRSmallConfig, TrainConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    model_config = RFDETRSmallConfig(num_classes=len(settings["class_names"]), **model_settings)
    train_config = TrainConfig(**settings)
    module = RFDETRModelModule(model_config, train_config)
    datamodule = RFDETRDataModule(model_config, train_config)
    trainer = build_trainer(train_config, model_config, accelerator="gpu", devices=1, fast_dev_run=8)
    trainer.fit(module, datamodule)


def train_detector(settings, model_settings, resume):
    from rfdetr import RFDETRSmall

    model = RFDETRSmall(**model_settings)
    model.train(
        **settings,
        device="cuda:0",
        resume=str(resume.resolve()) if resume else None,
        early_stopping=True,
        early_stopping_patience=5,
        checkpoint_interval=5,
        log_per_class_metrics=True,
    )


def evaluate_checkpoint(checkpoint, settings):
    from rfdetr import RFDETR

    model = RFDETR.from_checkpoint(str(checkpoint), device="cuda")
    return model.evaluate(
        dataset_dir=settings["dataset_dir"],
        split="val",
        device="cuda:0",
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        eval_max_dets=settings["eval_max_dets"],
        log_per_class_metrics=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/processed/rooms-expanded-coco")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts/runs/rooms-expanded-small")
    parser.add_argument("--weights", type=Path, required=True, help="Local COCO-pretrained Small weights")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.smoke and args.resume:
        parser.error("A smoke test starts from pretrained weights.")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch could not access an NVIDIA GPU.")
    if not args.weights.is_file():
        parser.error(f"Pretrained weights not found: {args.weights}")
    if not args.smoke and not args.resume and (args.output / "training_config.json").exists():
        parser.error(f"Training artifacts already exist in {args.output}. Use --resume or a fresh output.")
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    settings = training_settings(args)
    model_settings = {
        "pretrain_weights": str(args.weights.resolve()),
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    if args.smoke:
        run_smoke_test(settings, model_settings)
        return

    train_detector(settings, model_settings, args.resume)
    torch.cuda.empty_cache()
    metrics = evaluate_checkpoint(args.output / "checkpoint_best_total.pth", settings)
    report = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
    (args.output / "validation_metrics.json").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
