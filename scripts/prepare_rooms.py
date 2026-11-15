"""Convert the ADE20K Rooms splits to COCO object detection."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# These are complete ADE20K canonical names, including their embedded synonyms.
SOURCE_CLASSES = {
    "bathtub, bathing tub, bath, tub": "bathtub",
    "cabinet": "cabinet",
    "chair": "chair",
    "coffee maker": "coffee maker",
    "countertop": "countertop",
    "dishwasher, dish washer, dishwashing machine": "dishwasher",
    "faucet, spigot": "faucet",
    "floor, flooring": "floor",
    "microwave, microwave oven": "microwave",
    "mirror": "mirror",
    "oven": "oven",
    "hood, exhaust hood": "range hood",
    "refrigerator, icebox": "refrigerator",
    "shelf": "shelf",
    "shower": "shower",
    "shower door": "shower door",
    "shower stall, shower bath": "shower stall",
    "sink": "sink",
    "stove, kitchen stove, range, kitchen range, cooking stove": "stove",
    "table": "table",
    "toilet, can, commode, crapper, pot, potty, stool, throne": "toilet",
    "wall": "wall",
    "windowpane, window": "window",
}
VISIBLE_PIXELS = [0] * 255 + [255]


def read_visible_bounds(mask_path, image_size):
    with Image.open(mask_path) as mask:
        if mask.mode != "L" or mask.size != image_size:
            raise ValueError(f"Invalid ADE20K instance mask: {mask_path}")
        # ADE20K marks visible pixels as 255 and occluded pixels as 128.
        return mask.point(VISIBLE_PIXELS).getbbox()


def convert_image(annotation_path, source, image_id, category_ids):
    data = json.loads(annotation_path.read_bytes())["annotation"]
    image_path = annotation_path.with_suffix(".jpg")
    if image_path.name != data["filename"]:
        raise ValueError(f"Filename mismatch in {annotation_path}")
    with Image.open(image_path) as image:
        width, height = image.size

    image_record = {
        "id": image_id,
        "file_name": image_path.relative_to(source).as_posix(),
        "width": width,
        "height": height,
    }
    annotations = []
    fully_occluded = 0
    for obj in data["object"]:
        name = SOURCE_CLASSES.get(obj["name"])
        if name is None or obj["parts"]["part_level"] != 0:
            continue

        mask_path = annotation_path.parent / obj["instance_mask"].replace("\\", "/")
        bounds = read_visible_bounds(mask_path, (width, height))
        if bounds is None:
            fully_occluded += 1
            continue
        left, top, right, bottom = bounds
        box_width, box_height = right - left, bottom - top
        annotations.append(
            {
                "image_id": image_id,
                "category_id": category_ids[name],
                "bbox": [left, top, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            }
        )

    return image_record, annotations, fully_occluded


def convert_split(source, destination, category_ids, categories, *, require_all_classes=True):
    destination.mkdir()
    annotation_paths = sorted(source.glob("*/*.json"))
    if not annotation_paths:
        raise ValueError(f"No ADE20K annotations found in {source}")
    jpg_paths = sorted(source.glob("*/*.jpg"))
    if [path.with_suffix(".jpg") for path in annotation_paths] != jpg_paths:
        raise ValueError(f"JPEGs and annotation JSONs do not match in {source}")

    images = []
    annotations = []
    fully_occluded = 0
    for image_id, annotation_path in enumerate(annotation_paths, start=1):
        image, image_annotations, occluded = convert_image(annotation_path, source, image_id, category_ids)
        relative_path = image["file_name"]
        output_image = destination / relative_path
        output_image.parent.mkdir(exist_ok=True)
        shutil.copyfile(source / relative_path, output_image)
        images.append(image)
        fully_occluded += occluded
        for annotation_id, annotation in enumerate(image_annotations, start=len(annotations) + 1):
            annotations.append({"id": annotation_id, **annotation})

    class_names = {category_id: name for name, category_id in category_ids.items()}
    counts = Counter(class_names[annotation["category_id"]] for annotation in annotations)
    missing = category_ids.keys() - counts.keys()
    if missing and require_all_classes:
        raise ValueError(f"Classes with no boxes in {source}: {', '.join(sorted(missing))}")
    coco = {
        "info": {"description": "ADE20K Rooms bounding boxes"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    (destination / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    return {
        "images": len(images),
        "boxes": len(annotations),
        "fully_occluded_annotations": fully_occluded,
        "images_without_target_objects": len(images) - len({a["image_id"] for a in annotations}),
        "boxes_by_class": {name: counts[name] for name in sorted(category_ids)},
        "classes_without_boxes": sorted(missing),
    }


def export_dataset(source, output):
    category_ids = {name: i for i, name in enumerate(sorted(SOURCE_CLASSES.values()))}
    categories = [{"id": i, "name": name, "supercategory": "room object"} for name, i in category_ids.items()]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="rooms-coco-", dir=output.parent) as temp:
        staged = Path(temp) / "dataset"
        staged.mkdir()
        summary = {
            "train": convert_split(source / "train", staged / "train", category_ids, categories),
            "valid": convert_split(
                source / "validation", staged / "valid", category_ids, categories, require_all_classes=False
            ),
        }
        report = json.dumps(summary, indent=2) + "\n"
        (staged / "dataset_summary.json").write_text(report, encoding="utf-8")
        staged.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT / "data/raw/ade20k-rooms")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/processed/rooms-expanded-coco")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    print(export_dataset(args.source, args.output), end="")


if __name__ == "__main__":
    main()
