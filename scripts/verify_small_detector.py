"""Record local Small checkpoint compatibility and real-room inference evidence.

Run with the same Python environment as train_rooms.py. Inputs are local trusted
training artifacts; this script never downloads weights or modifies the run.
"""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

from PIL import Image, ImageOps

from prepare_rooms import PROJECT_ROOT, SOURCE_CLASSES


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(args):
    import supervision as sv
    import torch
    from rfdetr import RFDETR

    classes = sorted(SOURCE_CLASSES.values())
    coverage = {}
    for split in ("train", "valid"):
        path = args.dataset / split / "_annotations.coco.json"
        data = json.loads(path.read_bytes())
        expected = [(i, name) for i, name in enumerate(classes)]
        if sorted((c["id"], c["name"]) for c in data["categories"]) != expected:
            raise ValueError(f"Noncanonical category map: {path}")
        counts = Counter(a["category_id"] for a in data["annotations"])
        if set(counts) - set(range(len(classes))):
            raise ValueError(f"Unknown annotation category: {path}")
        if split == "train" and any(not counts[i] for i in range(len(classes))):
            raise ValueError("Every class requires training boxes")
        coverage[split] = {
            "annotations_sha256": sha256(path),
            "images": len(data["images"]),
            "boxes": len(data["annotations"]),
            "boxes_by_class": {name: counts[i] for i, name in enumerate(classes)},
        }

    run = args.checkpoint.parent
    config = json.loads((run / "training_config.json").read_bytes())
    if (config["model_config_type"] != "RFDETRSmallConfig"
            or config["class_names"] != classes
            or config["model_config"]["num_classes"] != 23
            or config["train_config"]["resume"] is not None):
        raise ValueError("Expected a fresh canonical 23-class Small run")
    model = RFDETR.from_checkpoint(str(args.checkpoint.resolve()), device=args.device)
    if type(model).__name__ != "RFDETRSmall" or list(model.class_names) != classes:
        raise ValueError("Loaded checkpoint architecture or classes differ from export")
    last = torch.load(run / "last.ckpt", map_location="cpu", weights_only=False)
    stop = next(v for k, v in last["callbacks"].items() if "EarlyStopping" in k)
    training = {
        "last_epoch_zero_based": last["epoch"],
        "global_step": last["global_step"],
        "stopped_epoch": stop["stopped_epoch"],
        "stopping_reason": stop["stopping_reason_message"],
    }
    del last
    args.output.mkdir(parents=True, exist_ok=True)
    images = []
    for index, path in enumerate(args.images):
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        detections = model.predict(image, threshold=args.threshold)
        labels = [f"{n} {c:.2f}" for n, c in zip(
            detections.data["class_name"], detections.confidence)]
        annotated = sv.BoxAnnotator().annotate(image.copy(), detections)
        annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
        preview = args.output / f"{index:02d}-{path.stem}.jpg"
        annotated.save(preview)
        images.append({
            "path": str(path), "sha256": sha256(path), "size": list(image.size),
            "preview": str(preview),
            "detections": [
                {"class_id": int(i), "class_name": str(n), "confidence": float(c),
                 "xyxy": box.tolist()}
                for i, n, c, box in zip(detections.class_id,
                    detections.data["class_name"], detections.confidence, detections.xyxy)
            ],
        })
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "architecture": type(model).__name__, "classes": classes,
        "threshold": args.threshold, "device": args.device,
        "platform": platform.platform(), "gpu": torch.cuda.get_device_name(0)
        if args.device == "cuda" else None,
        "versions": {n: importlib.metadata.version(n) for n in
                     ("rfdetr", "torch", "torchvision", "numpy", "pillow", "pytorch-lightning")},
        "training_config": config, "training_completion": training,
        "pretrained_sha256": sha256(Path(config["model_config"]["pretrain_weights"])),
        "coverage": coverage,
        "validation_metrics": json.loads((run / "validation_metrics.json").read_bytes()),
        "images": images,
        "limitations": ["Zero shower-door validation boxes: quality is unmeasurable.",
                         "Real photos are development evidence without ground truth.",
                         "Architecture choice does not demonstrate accuracy improvement."],
    }
    (args.output / "evidence.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Verified {len(classes)} classes and {len(images)} real images: {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT /
                        "artifacts/runs/rooms-expanded-small/checkpoint_best_total.pth")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data/processed/rooms-expanded-coco")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.output.exists():
        parser.error("Use a fresh evidence output directory")
    verify(args)


if __name__ == "__main__":
    main()
